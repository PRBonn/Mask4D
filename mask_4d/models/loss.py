# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former

from itertools import filterfalse

import torch
import torch.nn.functional as F
from mask_4d.utils.misc import pad_stack
from torch import nn
from torch.autograd import Variable


class MaskLoss(nn.Module):
    """The loss computation happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, data_cfg):
        """
        Parameters:
            num_classes: number of object categories
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_points: number of points to sample to compute the loss
        """
        super().__init__()
        self.num_classes = data_cfg.NUM_CLASSES
        self.ignore = data_cfg.IGNORE_LABEL

        self.weight_dict = {
            cfg.LOSS_WEIGHTS_KEYS[i]: cfg.LOSS_WEIGHTS[i]
            for i in range(len(cfg.LOSS_WEIGHTS))
        }

        self.eos_coef = cfg.EOS_COEF

        weights = torch.ones(self.num_classes + 1)
        weights[0] = 0.0
        weights[-1] = self.eos_coef
        self.weights = weights

        self.num_points = cfg.NUM_POINTS

    def forward(self, outputs, targets):
        pass

    def get_losses(
        self, outputs, targets, indices, num_masks, step="detect", skip=None
    ):
        classes = self.loss_classes(outputs, targets, indices, step, skip=skip)
        masks = self.loss_masks(outputs, targets, indices, num_masks, step)
        classes.update(masks)
        return classes

    def loss_classes(self, outputs, targets, indices, step="detect", skip=None):
        """Classification loss (NLL)
        targets dicts must contain the key "classes" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        pred_logits = outputs["pred_logits"].float()

        idx = self._get_pred_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["classes"], indices)]
        ).to(pred_logits.device)

        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,  # fill with class no_obj
            dtype=torch.int64,
            device=pred_logits.device,
        )
        target_classes[idx] = target_classes_o

        if skip is not None:
            # assign class 0 to ignore it when computing the loss
            target_classes[0][indices[0][0][skip]] = 0

        loss_ce = (
            F.cross_entropy(
                pred_logits.transpose(1, 2),
                target_classes,
                self.weights.to(pred_logits),
                ignore_index=self.ignore,
            )
            * self.weight_dict["loss_ce"]
        )
        losses = {"loss_ce" + step: loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, step="detect"):
        """Compute the losses related to the masks: the focal loss and the dice loss."""
        assert "pred_masks" in outputs

        masks = [t for t in targets["masks"]]
        n_masks = [m.shape[0] for m in masks]
        min_pts = min([m.shape[1] for m in masks])
        target_masks = pad_stack(masks)

        pred_idx = self._get_pred_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices, n_masks)
        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks[pred_idx[0], :, pred_idx[1]]
        target_masks = target_masks.to(pred_masks)
        target_masks = target_masks[tgt_idx]

        with torch.no_grad():
            # Random sampling num_points
            pt_idx = torch.randint(0, min_pts, (self.num_points, 1)).squeeze(1)
            point_labels = target_masks[:, pt_idx]

        point_logits = pred_masks[:, pt_idx]

        del pred_masks
        del target_masks

        losses = {
            "loss_mask"
            + step: sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
            * self.weight_dict["loss_mask"],
            "loss_dice"
            + step: dice_loss_jit(point_logits, point_labels, num_masks)
            * self.weight_dict["loss_dice"],
        }

        return losses

    def _get_pred_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices, n_masks):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        # From [B,id] to [id] of stacked masks
        cont_id = torch.cat(
            [torch.arange(n, device=tgt_idx.device) for n in n_masks],
        )
        b_id = torch.stack((batch_idx, cont_id), axis=1)
        map_m = torch.zeros((torch.max(batch_idx) + 1, max(n_masks)))
        for i in range(len(b_id)):
            map_m[b_id[i, 0], b_id[i, 1]] = i
        stack_ids = torch.tensor(
            [int(map_m[batch_idx[i], tgt_idx[i]]) for i in range(len(batch_idx))],
            device=tgt_idx.device,
        )
        return stack_ids


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    # loss = 1 - (numerator + 0.001) / (denominator + 0.001)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


class SemLoss(nn.Module):
    def __init__(self, w):
        super().__init__()

        self.ce_w, self.lov_w = w

        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, outputs, targets, n):
        ce = self.cross_entropy(outputs, targets)
        lovasz = self.lovasz_softmax(F.softmax(outputs, dim=1), targets)
        loss = {"sem_ce_" + n: self.ce_w * ce, "sem_lov_" + n: self.lov_w * lovasz}
        return loss

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax(self, probas, labels, classes="present", ignore=None):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        loss = self.lovasz_softmax_flat(
            *self.flatten_probas(probas, labels, ignore), classes=classes
        )
        return loss

    def lovasz_softmax_flat(self, probas, labels, classes="present"):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.0
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError("Sigmoid output possible only with 1 class")
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(
                torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted)))
            )
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        # Probabilities from SparseTensor.features already flattened
        N, C = probas.size()
        probas = probas.contiguous().view(-1, C)
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = labels != ignore
        vprobas = probas[torch.nonzero(valid).squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

    def isnan(self, x):
        return x != x

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = filterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == "raise":
                raise ValueError("Empty mean")
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n


class WrongLoss(nn.Module):
    def __init__(self, cfg, data_cfg):
        """
        Loss to account for correctly/incorrectly matched instances
        """
        super().__init__()
        self.weight_dict = {
            cfg.LOSS_WEIGHTS_KEYS[i]: cfg.LOSS_WEIGHTS[i]
            for i in range(len(cfg.LOSS_WEIGHTS))
        }

    def forward(self, track_ins, pred_idx, tgt_idx, targets, outputs, scan_i):
        """
        Arguments:
            track_ins: list of tracked instances
            pred_idx: matched predictions
            tgt_idx: matched ground truth
            targets: ground truth
            outputs:
            scan_i: number of the scan in the sequence
        Detection queries:
            * Matched GT is not a tracked instance, compute the loss.
            * Matched GT is a tracked instance, force to predict no_obj

        Tracking queries:
            * Correct instance matched, compute loss.
            * Incorrect instance matched, don't compute class loss and
              compute disimilarity loss
        """
        losses = {}
        tr_ids = track_ins.id[pred_idx]
        tgt_ids = targets["id"][0][tgt_idx]
        tracking_ids = torch.unique(track_ins.id)[1:]

        tracked = torch.isin(tgt_ids, tracking_ids)
        new_det = tr_ids == -1
        det_wrong_match = new_det * tracked
        det_wrong_idx = torch.nonzero(det_wrong_match).T[0]

        tr_q = tr_ids != -1
        unmatched_ids = tr_ids != tgt_ids
        tr_wrong_match = tr_q * unmatched_ids
        skip = tr_wrong_match
        tr_wrong_idx = torch.nonzero(tr_wrong_match).T[0]

        idx_delete = torch.cat((det_wrong_idx, tr_wrong_idx))

        if tr_wrong_match.sum() > 1:
            tr_wrong_pred_mask = outputs["pred_masks"][0][:, pred_idx[tr_wrong_match]]
            tr_wrong_tgt_mask = targets["masks"][0][tgt_idx[tr_wrong_match]]
            loss_wrong_mask = self.get_loss(
                tr_wrong_pred_mask, tr_wrong_tgt_mask, str(scan_i)
            )
            losses.update(loss_wrong_mask)
        return losses, idx_delete, skip

    def get_loss(self, pred_masks, tgt_masks, scan):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        The losses are computed for the wrongly matched pairs (tracked_ins,GT)
        The losses only consider the points which lay inside the GT masks and force
        those points in the predicted mask to have low logits.
        """
        # points inside GT masks should be 0
        gt_neg = tgt_masks - 1
        gt_neg[gt_neg < 0] = 1
        # tgt_masks will filter out the GT background points
        # from the loss computation

        loss_ce = (
            wrong_sigmoid_ce_loss_jit(pred_masks.T, gt_neg, tgt_masks.bool())
            * self.weight_dict["loss_wrong_mask"]
        )

        loss_dice = (
            wrong_dice_loss_jit(pred_masks.T, gt_neg, tgt_masks.bool())
            * self.weight_dict["loss_wrong_dice"]
        )

        losses = {
            "loss_wrong_mask_" + scan: loss_ce,
            "loss_wrong_dice_" + scan: loss_dice,
        }

        return losses


def wrong_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs[mask] * targets[mask]).sum(-1)
    denominator = inputs[mask].sum(-1) + targets[mask].sum(-1)
    # loss = 1 - (numerator + 0.001) / (denominator + 0.001)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


wrong_dice_loss_jit = torch.jit.script(wrong_dice_loss)  # type: torch.jit.ScriptModule


def wrong_sigmoid_ce_loss(
    inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    num_masks = len(mask)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # loss = (loss * mask).mean(1).sum() / num_masks
    loss = loss[mask].mean() / num_masks
    return loss


wrong_sigmoid_ce_loss_jit = torch.jit.script(
    wrong_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule
