import mask_4d.utils.misc as misc
import mask_4d.utils.testing as testing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mask_4d.models.backbone import SphericalEncoderDecoder
from mask_4d.models.decoder import MaskedTransformerDecoder
from mask_4d.models.loss import MaskLoss, SemLoss, WrongLoss
from mask_4d.models.matcher import HungarianMatcher
from mask_4d.models.positional_encoder import PositionalEncoder
from mask_4d.utils.evaluate_4dpanoptic import PanopticKitti4DEvaluator
from mask_4d.utils.evaluate_panoptic import PanopticKittiEvaluator
from mask_4d.utils.instances import Tracks
from mask_4d.utils.kalman_filter import KalmanBoxTracker
from pytorch_lightning.core.lightning import LightningModule


class Mask4D(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.cfg = hparams
        self.n_q = hparams.DECODER.NUM_QUERIES

        self.backbone = SphericalEncoderDecoder(hparams.BACKBONE, hparams.KITTI)
        self.decoder = MaskedTransformerDecoder(
            hparams.DECODER, hparams.BACKBONE, hparams.KITTI
        )

        self.mask_loss = MaskLoss(hparams.LOSS, hparams.KITTI)
        self.wrong_loss = WrongLoss(hparams.LOSS, hparams.KITTI)
        self.sem_loss = SemLoss(hparams.LOSS.SEM.WEIGHTS)

        self.evaluator = PanopticKittiEvaluator(hparams.KITTI)
        self.evaluator4d = PanopticKitti4DEvaluator(hparams.KITTI)

        self.matcher = HungarianMatcher(
            hparams.LOSS.LOSS_WEIGHTS, hparams.LOSS.NUM_POINTS
        )
        self.query_feat = nn.Embedding(
            self.n_q, hparams.DECODER.HIDDEN_DIM, device="cuda"
        )
        self.query_embed = nn.Embedding(
            self.n_q, hparams.DECODER.HIDDEN_DIM, device="cuda"
        )
        self.track_ins = self.init_tracks()
        self.last_ins_id = 1
        self.last_pose = np.eye(4)
        self.trackers = {}
        self.min_mask_pts = hparams.KITTI.MIN_POINTS

    def forward(self, x, track_ins):
        feats, coors, logits = self.backbone(x)
        outputs = self.decoder(feats, coors, track_ins)
        return outputs, logits

    def training_step(self, x: dict, idx):
        track_ins = self.init_tracks()
        losses = {}
        for s_i in range(len(x["pt_coord"])):
            scan = {k: [v[s_i]] for k, v in x.items()}
            outputs, logits = self.forward(scan, track_ins)
            track_ins.query = outputs["queries"][0]
            targets = {
                "classes": scan["masks_cls"],
                "masks": scan["masks"],
                "id": [scan["masks_ids"][0]],
            }
            matched_ins = []
            if (
                s_i != 0
                and len(track_ins) > self.n_q
                and (targets["id"][0] != 0).sum() > 0
            ):
                # Associate detections with previously tracked instances
                track_loss, matched_ins = self.tracking_step(
                    outputs, targets, track_ins, s_i
                )
                losses.update(track_loss)

            # Detect stuff and new instances
            pred_idx, tgt_idx, det_losses = self.detection_step(
                outputs, targets, matched_ins, track_ins, s_i
            )
            losses.update(det_losses)

            # aux losses
            losses.update(self.auxLoss(scan, logits, str(s_i)))

            track_ins = self.update_tr_instances(
                scan, outputs, pred_idx, tgt_idx, targets, track_ins
            )

            if s_i + 1 < len(x["pt_coord"]):
                track_ins.center = self.update_centers_ego(x, s_i, track_ins)

            new_track_ins = self.init_tracks()
            track_ins = Tracks.cat([new_track_ins, track_ins])

        # log losses
        loss_detect = 0
        loss_track = 0
        loss_wrong = 0
        for k, v in losses.items():
            if "detect" in k:
                loss_detect += v
            if "track" in k:
                loss_track += v
            if "wrong" in k:
                loss_wrong += v
        self.log(f"losses/detect", loss_detect, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log(f"losses/track", loss_track, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        self.log(f"losses/wrong", loss_wrong, batch_size=self.cfg.TRAIN.BATCH_SIZE)

        total_loss = sum(losses.values())
        self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        return total_loss

    def validation_step(self, x: dict, idx):
        self.evaluation_step(x, idx)

    def evaluation_step(self, x: dict, idx):
        sem_pred, ins_pred = self.panoptic4d_inference(x)
        self.evaluator.update(sem_pred, ins_pred, x)
        self.evaluator4d.update(sem_pred, ins_pred, x)

    def validation_epoch_end(self, outputs):
        self.log(
            "metrics/pq",
            self.evaluator.get_mean_pq(),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )
        self.log(
            "metrics/iou",
            self.evaluator.get_mean_iou(),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )
        self.log(
            "metrics/rq",
            self.evaluator.get_mean_rq(),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )
        self.evaluator4d.calculate_metrics()
        self.log(
            "metrics/aq",
            self.evaluator4d.get_mean_aq(),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )
        self.log(
            "metrics/pq4d",
            self.evaluator4d.get_mean_pq4d(),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
        )

        # reset tracking variables
        self.last_ins_id = 1
        self.track_ins = self.init_tracks()

        if not "EVALUATE" in self.cfg:
            self.evaluator.reset()
            self.evaluator4d.reset()

    def test_step(self, x: dict, idx):
        _sem_pred, _ins_pred = self.panoptic4d_inference(x)
        sem_pred, ins_pred = testing.full_preds(x, _sem_pred, _ins_pred)
        results_dir = self.cfg.RESULTS_DIR
        class_inv_lut = self.evaluator.get_class_inv_lut()
        testing.save_results(sem_pred, ins_pred, results_dir, x, class_inv_lut)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.TRAIN.STEP, gamma=self.cfg.TRAIN.DECAY
        )
        return [optimizer], [scheduler]

    def panoptic4d_inference(self, x):
        sem_pred = []
        ins_pred = []
        for i in range(len(x["pt_coord"])):
            if int(x["fname"][i].split("/")[-1][:-4]) == 0:
                self.track_ins = self.init_tracks()
                self.last_ins_id = 1
                self.last_pose = x["pose"][i]
            elif len(self.track_ins) > self.n_q:
                self.update_centers_motion(x, i)

            scan = {k: [v[i]] for k, v in x.items()}
            outputs, logits = self.forward(scan, self.track_ins)
            self.track_ins.query = outputs["queries"][0]

            sem_pred.append(self.semantic_predictions(outputs))
            ins_pred.append(self.instance_predictions(outputs, scan))

            self.update_motion_model()

            new_track_ins = self.init_tracks()
            self.track_ins = Tracks.cat([new_track_ins, self.track_ins])
        return sem_pred, ins_pred

    def semantic_predictions(self, outputs):
        mask_cls = outputs["pred_logits"][0]
        mask_pred = outputs["pred_masks"][0]
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        pred = torch.einsum("qc,pq->pc", mask_cls, mask_pred)
        semseg = torch.argmax(pred, dim=1)
        return semseg.cpu().numpy()

    def instance_predictions(self, outputs, x):
        points = torch.from_numpy(x["pt_coord"][0]).to(outputs["pred_masks"])
        pose = x["pose"][0]
        glob_points = misc.apply_pose(points, pose)
        ins_pred = torch.zeros(points.shape[0]).long()
        mask_cls = outputs["pred_logits"][0]
        mask_pred = outputs["pred_masks"][0].sigmoid()
        things_ids = torch.tensor(self.trainer.datamodule.things_ids).cuda()
        num_classes = self.cfg.KITTI.NUM_CLASSES

        scores, labels = mask_cls.max(-1)
        keep = labels.ne(num_classes) & torch.isin(labels, things_ids)

        if keep.sum() == 0:  # no detected instances
            self.track_ins = self.track_ins[self.track_ins.id != -1]
            self.track_ins.life -= 1
            self.track_ins = self.track_ins[self.track_ins.life > 0]
            return ins_pred.cpu().numpy()

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[:, keep]
        cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
        cur_mask_ids = cur_prob_masks.argmax(1)
        cur_ins_ids = self.track_ins.id[keep]
        ins_idx = torch.where(keep)[0]

        for k in range(cur_classes.shape[0]):
            mask_area = (cur_mask_ids == k).sum().item()  # points in mask k
            original_area = (cur_masks[:, k] >= 0.5).sum().item()  # binary mask
            mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
            ins_id = cur_ins_ids[k]

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < self.cfg.MODEL.OVERLAP_THRESHOLD:
                    continue
                if mask.sum() < self.min_mask_pts:
                    continue

                self.track_ins.life[ins_idx[k]] += 1
                pts = misc.remove_outliers(points[mask])
                if pts.shape[0] < self.min_mask_pts:
                    continue
                g_pts = misc.remove_outliers(glob_points[mask])
                size, angle = misc.fit_ellipse(pts)
                self.track_ins.size_xy[ins_idx[k]] = size
                self.track_ins.angle[ins_idx[k]] = angle
                self.track_ins.bbox[ins_idx[k]] = misc.bbox_from_pts(g_pts)
                self.track_ins.center[ins_idx[k]] = misc.center_from_points(pts)

                if ins_id == -1:  # if new detected instance
                    self.track_ins.id[ins_idx[k]] = self.last_ins_id
                    ins_id = self.last_ins_id
                    self.last_ins_id += 1
                # assign instance ID to points
                ins_pred[mask == 1] = ins_id
        self.track_ins = self.track_ins[self.track_ins.id != -1]
        self.track_ins.life -= 1
        self.track_ins = self.track_ins[self.track_ins.life > 0]

        return ins_pred.cpu().numpy()

    def init_tracks(self):
        """
        Init list of N tracks with the following params:
            query: [N,C] query feature to decode the mask
            query_pe: [N,C] query positional embedding
            id: [N,1] track identifier
            life: [N,1] for how many steps keep the track alive
            center: [N,3] center of the mask
            size_xy: [N,2,3] size of the instance in X and Y
            angle: [N,1] orientation of the instance (theta)
            bbox: [N,7]: [x,y,z,theta,l,w,h] 3D bounding box around the instance
        """
        tracks = Tracks()
        N = self.query_feat.weight.shape[0]
        dev = self.query_feat.weight.device
        tracks.query = self.query_feat.weight
        tracks.query_pe = self.query_embed.weight
        tracks.id = torch.full((N,), -1, device=dev)
        tracks.life = torch.full((N,), self.cfg.MODEL.TR_LIFE, device=dev)
        tracks.center = torch.full((N, 3), 0.0, device=dev)
        tracks.size_xy = torch.full((N, 3), 0.0, device=dev)
        tracks.angle = torch.full((N,), 0.0, device=dev)
        tracks.bbox = torch.full((N, 7), 0.0, device=dev)
        return tracks

    def detection_step(self, outputs, targets, matched_ins, track_ins, scan_i):
        losses = {}
        indices = self.matcher(outputs, targets)

        pred_idx = self.mask_loss._get_pred_permutation_idx(indices)[1]
        n_masks = [m.shape[0] for m in targets["masks"]]
        tgt_idx = self.mask_loss._get_tgt_permutation_idx(indices, n_masks)

        skip = None
        if len(matched_ins) > 0:
            loss_wrong, idx_delete, skip = self.wrong_loss(
                track_ins, pred_idx, tgt_idx, targets, outputs, scan_i
            )
            losses.update(loss_wrong)
            # delete from predictions and targets
            pred_idx = misc.th_delete(pred_idx, idx_delete)
            tgt_idx = misc.th_delete(tgt_idx, idx_delete)

        # compute loss on pairs matched with Hungarian
        det_loss = self.mask_loss.get_losses(
            outputs,
            targets,
            indices,
            len(indices[0][0]),
            "_detect_" + str(scan_i),
            skip=skip,
        )
        det_loss = {k: self.cfg.LOSS.DET_W * v for k, v in det_loss.items()}
        losses.update(det_loss)
        losses.update(self.aux_outs_loss(outputs, targets, indices, scan_i))
        return pred_idx, tgt_idx, losses

    def tracking_step(self, outputs, targets, track_ins, scan_i):
        matched_ins, matched_outputs, matched_tgt = self.fixed_match(
            track_ins, outputs, targets
        )

        track_loss = {}
        if len(matched_ins) > 0:
            # create indices for the loss function
            idx = torch.arange(len(matched_ins), device=matched_ins.query.device)
            matched_idx = [(idx, idx)]
            # compute loss on fixed matched pairs
            track_loss = self.mask_loss.get_losses(
                matched_outputs,
                matched_tgt,
                matched_idx,
                len(matched_ins),
                "_track_" + str(scan_i),
            )
            track_loss = {k: self.cfg.LOSS.TRACK_W * v for k, v in track_loss.items()}
        return track_loss, matched_ins

    def fixed_match(self, track_ins, outputs, targets):
        """
        Fixed match between tracked instances and targets based on ID.
        Return matched and non-matched instances, outputs and targets
        """
        m_ins_ids, m_tgt_iids = torch.where(track_ins.id[:, None] == targets["id"][0])
        m_out_log = outputs["pred_logits"][:, m_ins_ids, :]
        m_out_masks = outputs["pred_masks"][:, :, m_ins_ids]
        m_tgt_classes = targets["classes"][0][m_tgt_iids]
        m_tgt_masks = targets["masks"][0][m_tgt_iids]
        m_tgt_ids = targets["id"][0][m_tgt_iids]
        if len(m_ins_ids) == 0:
            return [], [], [], [], []

        matched_outputs = {"pred_logits": m_out_log, "pred_masks": m_out_masks}
        matched_targets = {
            "classes": [m_tgt_classes],
            "masks": [m_tgt_masks],
            "id": [m_tgt_ids],
        }
        matched_ins = track_ins[m_ins_ids]

        return matched_ins, matched_outputs, matched_targets

    def update_tr_instances(self, scan, outputs, pred_idx, tgt_idx, targets, track_ins):
        """
        Update tracked instances parameters: angle, size, center and id.
        The ids are according to associations made by the Hungarian matcher.
        """
        for p in range(len(pred_idx)):
            if targets["id"][0][tgt_idx[p]] > 0:  # ignore stuff
                coors = torch.from_numpy(scan["pt_coord"][0]).cuda()
                _mask = (outputs["pred_masks"][0, :, pred_idx[p]]).sigmoid() > 0.5
                pts = coors[_mask]
                if pts.shape[0] <= self.min_mask_pts:
                    continue
                track_ins.id[pred_idx[p]] = targets["id"][0][tgt_idx[p]]
                size, angle = misc.fit_ellipse(pts)
                track_ins.size_xy[pred_idx[p]] = size
                track_ins.angle[pred_idx[p]] = angle
                track_ins.center[pred_idx[p]] = misc.center_from_points(pts)
        # remove track_ins with id=-1 (out queries not matched with GT instance)
        track_ins = track_ins[track_ins.id != -1]
        return track_ins

    def update_centers_ego(self, x, scan_i, track_ins):
        """
        Update instance centers compensating the ego motion
        """
        cent = track_ins.center
        p0 = x["pose"][scan_i]
        p1 = x["pose"][scan_i + 1]
        t0_1 = np.linalg.inv(p1) @ p0
        center = misc.apply_pose(cent, t0_1)
        return center

    def update_centers_motion(self, x, i):
        # update centers using the motion model
        inv_p1 = np.linalg.inv(x["pose"][i])
        for tr in self.trackers:
            self.trackers[tr].predict()
        for j in range(len(self.track_ins.center[self.n_q :])):
            id = self.track_ins.id[self.n_q + j].item()
            pred_center = torch.from_numpy(
                self.trackers[id].get_state()[:3][None, :]
            ).cuda()
            # transform from global to current frame
            local_center = misc.apply_pose(pred_center, inv_p1)[0]
            self.track_ins.center[self.n_q + j] = local_center

    def update_motion_model(self):
        for id, bbox in zip(self.track_ins.id, self.track_ins.bbox):
            if id.item() not in self.trackers.keys():
                self.trackers[id.item()] = KalmanBoxTracker(bbox.cpu())
            else:
                self.trackers[id.item()].update(bbox.cpu())

    def aux_outs_loss(self, outputs, targets, indices, scan_i):
        all_losses = {}
        for aux_i, aux_outputs in enumerate(outputs["aux_outputs"]):
            # detection aux losses
            aux_indices = self.matcher(aux_outputs, targets)
            l_dict = self.mask_loss.get_losses(
                aux_outputs,
                targets,
                aux_indices,
                len(indices[0][0]),
                "_detect_" + str(scan_i),
            )
            l_dict = {
                f"{aux_i}_" + k: self.cfg.LOSS.DET_W_AUX * v for k, v in l_dict.items()
            }
            all_losses.update(l_dict)
        return all_losses

    def auxLoss(self, x, bb_log, n):
        loss = {}
        labels = torch.from_numpy(x["sem_label"][0]).type(torch.LongTensor).cuda()[:, 0]
        loss_sem_bb = self.sem_loss(bb_log, labels, n)
        loss.update(loss_sem_bb)
        return loss
