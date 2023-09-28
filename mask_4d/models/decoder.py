# Modified from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
import mask_4d.models.blocks as blocks
import mask_4d.utils.misc as misc
import torch
from mask_4d.models.positional_encoder import PositionalEncoder
from torch import nn


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, cfg, bb_cfg, data_cfg):
        super().__init__()
        hidden_dim = cfg.HIDDEN_DIM
        self.cfg = cfg
        self.pe_layer = PositionalEncoder(cfg.POS_ENC)
        self.num_layers = cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS
        self.nheads = cfg.NHEADS
        self.num_queries = cfg.NUM_QUERIES
        self.num_feature_levels = cfg.FEATURE_LEVELS

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.mask_feat_proj = nn.Sequential()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                blocks.SelfAttentionLayer(hidden_dim, self.nheads)
            )
            self.transformer_cross_attention_layers.append(
                blocks.PositionCrossAttentionLayer(hidden_dim, self.nheads)
            )

            self.transformer_ffn_layers.append(
                blocks.FFNLayer(hidden_dim, cfg.DIM_FEEDFORWARD)
            )

        if bb_cfg.CHANNELS[0] != hidden_dim:
            self.mask_feat_proj = nn.Linear(bb_cfg.CHANNELS[0], hidden_dim)
        in_channels = self.num_feature_levels * [bb_cfg.CHANNELS[0]]

        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != hidden_dim:  # linear projection to hidden_dim
                self.input_proj.append(nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(nn.Sequential())

        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        self.mask_embed = blocks.MLP(hidden_dim, hidden_dim, hidden_dim, 3)

    def forward(self, feats, coors, track_ins):
        mask_features = self.mask_feat_proj(feats) + self.pe_layer(coors)
        src = []
        pos = []
        queries = []

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(coors))
            feat = self.input_proj[i](feats)
            src.append(feat)

        query_embed = track_ins.query_pe.unsqueeze(0)
        output = track_ins.query.unsqueeze(0)
        q_centers = track_ins.center
        size = track_ins.size_xy
        angle = track_ins.angle

        predictions_class = []
        predictions_mask = []

        if output.shape[-1] > self.num_queries:
            mask_kernel = self.compute_kernel(q_centers, coors, size, angle, mask=True)
        else:
            mask_kernel = torch.zeros((1, coors.shape[1], q_centers.shape[0])).to(
                output.device
            )

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, mask_kernel
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        queries.append(output)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if attn_mask is not None:
                attn_mask[attn_mask.sum(-1) == attn_mask.shape[-1]] = False

            kernel = self.compute_kernel(q_centers, coors, size, angle)

            output = self.transformer_cross_attention_layers[i](
                output,
                kernel,
                src[level_index],
                attn_mask=attn_mask,
                padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, attn_mask=None, padding_mask=None, query_pos=query_embed
            )

            output = self.transformer_ffn_layers[i](output)

            if output.shape[-1] > self.num_queries:
                mask_kernel = self.compute_kernel(
                    q_centers, coors, size, angle, mask=True
                )
            else:
                mask_kernel = torch.zeros_like(outputs_mask)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, mask_kernel
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            queries.append(output)

        assert len(predictions_class) == self.num_layers + 1
        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "queries": queries[-1],
        }

        out["aux_outputs"] = self.set_aux(predictions_class, predictions_mask, queries)

        return out

    def forward_prediction_heads(
        self,
        output,
        mask_features,
        kernel,
    ):
        decoder_output = self.decoder_norm(output)  # Layer norm
        outputs_class = self.class_embed(decoder_output)  # Linear

        mask_embed = self.mask_embed(decoder_output)  # MLP

        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)

        attn_mask = outputs_mask.sigmoid() + kernel
        attn_mask = attn_mask - attn_mask.min()
        attn_mask = attn_mask / attn_mask.max()
        attn_mask = (attn_mask < 0.5).detach().bool()

        # Create binary mask
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.nheads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def set_aux(self, outputs_class, outputs_seg_masks, queries):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b, "queries": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], queries[:-1])
        ]

    def compute_kernel(self, q_centers, coors, size, angle, mask=False):
        dx = torch.cdist(
            q_centers[self.num_queries :][:, 0].unsqueeze(1), coors[:, :, 0].T
        )
        dy = torch.cdist(
            q_centers[self.num_queries :][:, 1].unsqueeze(1), coors[:, :, 1].T
        )

        sx = size[self.num_queries :, 0]
        sy = size[self.num_queries :, 1]
        angles = angle[self.num_queries :]

        x_mu = torch.cat((dx.unsqueeze(-1), dy.unsqueeze(-1)), dim=-1)
        covs = misc.getcovs(sx, sy, angles)
        inv_covs = torch.linalg.inv(covs)
        k_weights = -0.5 * torch.einsum("bnd,bdd,bnd->bn", x_mu, inv_covs, x_mu)
        if mask:
            k_weights = torch.exp(k_weights)
        kernel = torch.zeros((1, coors.shape[1], q_centers.shape[0])).to(
            q_centers.device
        )
        kernel[0, :, self.num_queries :] = k_weights.T
        return kernel
