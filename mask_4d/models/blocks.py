from typing import Optional

import torch
from mask_4d.models.position_attention import PositionAttention
from torch import Tensor, nn
from torch.nn import functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, pre_norm=False, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        q_embed,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.pre_norm:
            q_embed = self.norm(q_embed)
            q = k = self.with_pos_embed(q_embed, query_pos)
            q_embed2 = self.self_attn(
                q, k, value=q_embed, attn_mask=attn_mask, key_padding_mask=padding_mask
            )[0]
            q_embed = q_embed + self.dropout(q_embed2)
        else:
            q = k = self.with_pos_embed(q_embed, query_pos)
            q_embed2 = self.self_attn(
                q, k, value=q_embed, attn_mask=attn_mask, key_padding_mask=padding_mask
            )[0]
            q_embed = q_embed + self.dropout(q_embed2)
            q_embed = self.norm(q_embed)
        return q_embed


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, pre_norm=False, activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def with_pos_embed2(self, tensor, pos: Optional[Tensor]):
        out = torch.cat((tensor, pos.unsqueeze(0)), dim=-1)
        return out

    def forward(
        self,
        q_embed,
        bb_feat,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.pre_norm:
            q_embed2 = self.multihead_attn(
                query=self.with_pos_embed(q_embed, query_pos),
                key=self.with_pos_embed(bb_feat, pos),
                value=self.with_pos_embed(bb_feat, pos),
                # value=bb_feat,
                attn_mask=attn_mask,
                key_padding_mask=padding_mask,
            )[0]
            q_embed = q_embed + self.dropout(q_embed2)
            q_embed = self.norm(q_embed)
        else:
            q_embed = self.norm(q_embed)
            q_embed2 = self.multihead_attn(
                query=self.with_pos_embed(q_embed, query_pos),
                key=self.with_pos_embed(bb_feat, pos),
                value=self.with_pos_embed(bb_feat, pos),
                # value=bb_feat,
                attn_mask=attn_mask,
                key_padding_mask=padding_mask,
            )[0]
            q_embed = q_embed + self.dropout(q_embed2)
        return q_embed


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        pre_norm=False,
        activation="relu",
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt):
        if self.pre_norm:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
        else:
            tgt = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout(tgt2)
        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
    ):
        super().__init__()
        self.multihead_attn = PositionAttention(d_model, nhead, dropout=dropout)

        self.nhead = nhead

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        q_embed,
        kernel,
        bb_feat,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q_embed = self.norm(q_embed)
        q_embed2 = self.multihead_attn(
            query=self.with_pos_embed(q_embed, query_pos),
            key=self.with_pos_embed(bb_feat, pos),
            value=self.with_pos_embed(bb_feat, pos),
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            kernel=kernel,
        )
        q_embed = q_embed + self.dropout(q_embed2)
        return q_embed
