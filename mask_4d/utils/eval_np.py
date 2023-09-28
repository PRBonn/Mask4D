#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import math

import numpy as np


class PanopticEval:
    """Panoptic evaluation using numpy

    authors: Andres Milioto and Jens Behley

    """

    def __init__(
        self, n_classes, device=None, ignore=None, offset=2**32, min_points=30
    ):
        self.n_classes = n_classes
        assert device == None
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64
        )

        # print("[PANOPTIC EVAL] IGNORE: ", self.ignore)
        # print("[PANOPTIC EVAL] INCLUDE: ", self.include)

        self.reset()
        self.offset = offset  # largest number of instances in a given scan
        self.min_points = (
            min_points  # smallest number of points to consider instances in gt
        )
        self.eps = 1e-15

    def num_classes(self):
        return self.n_classes

    def merge(self, evaluator):
        self.px_iou_conf_matrix += evaluator.px_iou_conf_matrix
        self.pan_tp += evaluator.pan_tp
        self.pan_iou += evaluator.pan_iou
        self.pan_fp += evaluator.pan_fp
        self.pan_fn += evaluator.pan_fn

        self.evaluated_fnames += evaluator.evaluated_fnames

    def reset(self):
        # general things
        # iou stuff
        self.px_iou_conf_matrix = np.zeros(
            (self.n_classes, self.n_classes), dtype=np.int64
        )
        # panoptic stuff
        self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
        self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

        self.evaluated_fnames = []

    ################################# IoU STUFF ##################################
    def addBatchSemIoU(self, x_sem, y_sem):
        # idxs are labels and predictions
        idxs = np.stack([x_sem, y_sem], axis=0)

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def getSemIoUStats(self):
        # clone to avoid modifying the real deal
        conf = self.px_iou_conf_matrix.copy().astype(np.double)
        # remove fp from confusion on the ignore classes predictions
        # points that were predicted of another class, but were ignore
        # (corresponds to zeroing the cols of those classes, since the predictions
        # go on the rows)
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getSemIoU(self):
        tp, fp, fn = self.getSemIoUStats()
        # print(f"tp={tp}")
        # print(f"fp={fp}")
        # print(f"fn={fn}")
        intersection = tp
        union = tp + fp + fn
        union = np.maximum(union, self.eps)
        iou = intersection.astype(np.double) / union.astype(np.double)
        iou_mean = (
            intersection[self.include].astype(np.double)
            / union[self.include].astype(np.double)
        ).mean()

        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getSemAcc(self):
        tp, fp, fn = self.getSemIoUStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum()
        total = np.maximum(total, self.eps)
        acc_mean = total_tp.astype(np.double) / total.astype(np.double)

        return acc_mean  # returns "acc mean"

    ################################# IoU STUFF ##################################
    ##############################################################################

    #############################  Panoptic STUFF ################################
    def addBatchPanoptic(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
        # make sure instances are not zeros (it messes with my approach)
        x_inst_row = x_inst_row + 1
        y_inst_row = y_inst_row + 1

        # only interested in points that are outside the void area (not in excluded classes)
        for cl in self.ignore:
            # make a mask for this class
            gt_not_in_excl_mask = y_sem_row != cl
            # remove all other points
            x_sem_row = x_sem_row[gt_not_in_excl_mask]
            y_sem_row = y_sem_row[gt_not_in_excl_mask]
            x_inst_row = x_inst_row[gt_not_in_excl_mask]
            y_inst_row = y_inst_row[gt_not_in_excl_mask]

        # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
        for cl in self.include:
            # print("*"*80)
            # print("CLASS", cl.item())
            # get a class mask
            x_inst_in_cl_mask = x_sem_row == cl
            y_inst_in_cl_mask = y_sem_row == cl

            # get instance points in class (makes outside stuff 0)
            x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

            # generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(
                x_inst_in_cl[x_inst_in_cl > 0], return_counts=True
            )
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])
            # print("Unique predictions:", unique_pred)

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(
                y_inst_in_cl[y_inst_in_cl > 0], return_counts=True
            )
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])
            # print("Unique ground truth:", unique_gt)

            # generate intersection using offset
            valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
            offset_combo = (
                x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
            )
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            # generate an intersection map
            # count the intersections with over 0.5 IoU as TP
            gt_labels = unique_combo // self.offset
            pred_labels = unique_combo % self.offset
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(float) / unions.astype(float)

            tp_indexes = ious > 0.5
            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

            # count the FN
            self.pan_fn[cl] += np.sum(
                np.logical_and(counts_gt >= self.min_points, matched_gt == False)
            )

            # count the FP
            self.pan_fp[cl] += np.sum(
                np.logical_and(counts_pred >= self.min_points, matched_pred == False)
            )

    def getPQ(self):
        # first calculate for all classes
        sq_all = self.pan_iou.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double), self.eps
        )
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double)
            + 0.5 * self.pan_fp.astype(np.double)
            + 0.5 * self.pan_fn.astype(np.double),
            self.eps,
        )
        pq_all = sq_all * rq_all

        # then do the REAL mean (no ignored classes)
        SQ = sq_all[self.include].mean()
        RQ = rq_all[self.include].mean()
        PQ = pq_all[self.include].mean()

        return PQ, SQ, RQ, pq_all, sq_all, rq_all

    #############################  Panoptic STUFF ################################
    ##############################################################################

    def addBatch(self, x_sem, x_inst, y_sem, y_inst):  # x=preds, y=targets
        """IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]"""
        # add to IoU calculation (for checking purposes)
        self.addBatchSemIoU(x_sem, y_sem)

        # now do the panoptic stuff
        self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst)

    def addBatch_w_fname(
        self, x_sem, x_inst, y_sem, y_inst, fname
    ):  # x=preds, y=targets
        """IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]"""
        # add to IoU calculation (for checking purposes)
        self.addBatchSemIoU(x_sem, y_sem)

        # now do the panoptic stuff
        self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst)

        self.evaluated_fnames.append(fname)


class Panoptic4DEval:
    """Panoptic evaluation using numpy

    authors: Andres Milioto, Jens Behley, Aljosa Osep

    """

    def __init__(
        self, n_classes, device=None, ignore=None, offset=2**32, min_points=30
    ):
        self.n_classes = n_classes
        # assert (device == None)
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64
        )
        # print("[PANOPTIC4D EVAL] IGNORE: ", self.ignore)
        # print("[PANOPTIC4D EVAL] INCLUDE: ", self.include)

        self.reset()
        self.offset = offset  # largest number of instances in a given scan
        self.min_points = (
            min_points  # smallest number of points to consider instances in gt
        )
        self.eps = 1e-15

    def num_classes(self):
        return self.n_classes

    def reset(self):
        # general things
        # iou stuff
        self.px_iou_conf_matrix = np.zeros(
            (self.n_classes, self.n_classes), dtype=np.int64
        )

        self.sequences = []
        self.preds = {}
        self.gts = {}
        self.intersects = {}
        self.intersects_ovr = {}

        # Per-class association quality collect here
        self.pan_aq = np.zeros(self.n_classes, dtype=np.double)
        self.pan_aq_ovr = 0.0

    ################################# IoU STUFF ##################################
    def addBatchSemIoU(self, x_sem, y_sem):
        # idxs are labels and predictions
        idxs = np.stack([x_sem, y_sem], axis=0)

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def getSemIoUStats(self):
        # clone to avoid modifying the real deal
        conf = self.px_iou_conf_matrix.copy().astype(np.double)
        # remove fp from confusion on the ignore classes predictions
        # points that were predicted of another class, but were ignore
        # (corresponds to zeroing the cols of those classes, since the predictions
        # go on the rows)
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getSemIoU(self):
        tp, fp, fn = self.getSemIoUStats()
        # print(f"tp={tp}")
        # print(f"fp={fp}")
        # print(f"fn={fn}")
        intersection = tp
        union = tp + fp + fn

        num_present_classes = np.count_nonzero(union)

        union = np.maximum(union, self.eps)
        iou = intersection.astype(np.double) / union.astype(np.double)
        iou_mean = np.sum(iou) / num_present_classes

        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getSemAcc(self):
        tp, fp, fn = self.getSemIoUStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum()
        total = np.maximum(total, self.eps)
        acc_mean = total_tp.astype(np.double) / total.astype(np.double)

        return acc_mean  # returns "acc mean"

    ################################# IoU STUFF ##################################
    ##############################################################################

    #############################  Panoptic STUFF ################################

    def update_dict_stat(self, stat_dict, unique_ids, unique_cnts):
        for uniqueid, counts in zip(unique_ids, unique_cnts):
            if uniqueid == 1:
                continue  # 1 -- no instance
            if uniqueid in stat_dict:
                stat_dict[uniqueid] += counts
            else:
                stat_dict[uniqueid] = counts

    def addBatchPanoptic4D(self, seq, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
        # start = time.time()
        if seq not in self.sequences:
            self.sequences.append(seq)
            self.preds[seq] = {}
            self.gts[seq] = [{} for i in range(self.n_classes)]
            self.intersects[seq] = [{} for i in range(self.n_classes)]
            self.intersects_ovr[seq] = [{} for i in range(self.n_classes)]

        # make sure instances are not zeros (it messes with my approach)
        x_inst_row = x_inst_row + 1
        y_inst_row = y_inst_row + 1

        # only interested in points that are outside the void area (not in excluded classes)
        for cl in self.ignore:
            # make a mask for this class
            gt_not_in_excl_mask = y_sem_row != cl
            # remove all other points
            x_sem_row = x_sem_row[gt_not_in_excl_mask]
            y_sem_row = y_sem_row[gt_not_in_excl_mask]
            x_inst_row = x_inst_row[gt_not_in_excl_mask]
            y_inst_row = y_inst_row[gt_not_in_excl_mask]

        for cl in self.include:
            # print("*"*80)
            # print("CLASS", cl.item())

            # Per-class accumulated stats
            cl_preds = self.preds[seq]
            cl_gts = self.gts[seq][cl]
            cl_intersects = self.intersects[seq][cl]

            # get a binary class mask (filter acc. to semantic class!)
            x_inst_in_cl_mask = x_sem_row == cl
            y_inst_in_cl_mask = y_sem_row == cl

            # get instance points in class (mask-out everything but _this_ class)
            x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

            # generate the areas for each unique instance gt_np (i.e., set2)
            unique_gt, counts_gt = np.unique(
                y_inst_in_cl[y_inst_in_cl > 0], return_counts=True
            )
            self.update_dict_stat(
                cl_gts,
                unique_gt[counts_gt > self.min_points],
                counts_gt[counts_gt > self.min_points],
            )

            valid_combos_min_point = np.zeros_like(
                y_inst_in_cl
            )  # instances which have at least self.min points
            for valid_id in unique_gt[counts_gt > self.min_points]:
                valid_combos_min_point = np.logical_or(
                    valid_combos_min_point, y_inst_in_cl == valid_id
                )

            y_inst_in_cl = y_inst_in_cl * valid_combos_min_point
            # generate the areas for each unique instance prediction (i.e., set1)
            unique_pred, counts_pred = np.unique(
                x_inst_in_cl[x_inst_in_cl > 0], return_counts=True
            )

            # is there better way to do this?
            self.update_dict_stat(cl_preds, unique_pred, counts_pred)

            valid_combos = np.logical_and(
                x_inst_row > 0, y_inst_in_cl > 0
            )  # Convert to boolean and do logical and, based on semantics

            # generate intersection using offset
            offset_combo = (
                x_inst_row[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
            )
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            self.update_dict_stat(cl_intersects, unique_combo, counts_combo)

    def getPQ4D(self):
        num_tubes = [0] * self.n_classes
        for seq in self.sequences:
            for cl in self.include:
                cl_preds = self.preds[seq]
                cl_gts = self.gts[seq][cl]
                cl_intersects = self.intersects[seq][cl]
                outer_sum = 0.0
                outer_sum_iou = 0.0
                num_tubes[cl] += len(cl_gts)
                for gt_id, gt_size in cl_gts.items():
                    inner_sum = 0.0
                    inner_sum_iou = 0.0
                    for pr_id, pr_size in cl_preds.items():
                        # TODO: pay attention for zero intersection!
                        TPA_key = pr_id + self.offset * gt_id
                        if TPA_key in cl_intersects:
                            TPA = cl_intersects[TPA_key]
                            Prec = TPA / float(
                                pr_size
                            )  # TODO I dont think these can ever be zero, but double check
                            Recall = TPA / float(gt_size)
                            TPA_ovr = self.intersects[seq][cl][TPA_key]
                            inner_sum_iou += TPA_ovr * (
                                TPA_ovr / (gt_size + pr_size - TPA_ovr)
                            )
                            if Prec > 1.0 or Recall > 1.0:
                                print("something wrong !!")
                    outer_sum_iou += 1.0 / float(gt_size) * float(inner_sum_iou)
                self.pan_aq[
                    cl
                ] += outer_sum_iou  # 1.0/float(len(cl_gts)) # Normalize by #tubes
                self.pan_aq_ovr += outer_sum_iou
        # ==========

        AQ_overall = np.sum(self.pan_aq_ovr) / np.sum(num_tubes[1:9])
        AQ = self.pan_aq / np.maximum(num_tubes, self.eps)

        iou_mean, iou = self.getSemIoU()

        PQ4D = math.sqrt(AQ_overall * iou_mean)
        return PQ4D, AQ_overall, AQ, iou, iou_mean

    #############################  Panoptic STUFF ################################
    ##############################################################################

    def addBatch(self, seq, x_sem, x_inst, y_sem, y_inst):  # x=preds, y=targets
        """IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]"""
        # add to IoU calculation (for checking purposes)
        self.addBatchSemIoU(x_sem, y_sem)

        # now do the panoptic stuff
        self.addBatchPanoptic4D(seq, x_sem, x_inst, y_sem, y_inst)


def gen_psuedo_labels(n=50):
    # generate ground truth and prediction
    sem_pred = []
    inst_pred = []
    sem_gt = []
    inst_gt = []

    # some ignore stuff
    N_ignore = n
    sem_pred.extend([0 for i in range(N_ignore)])
    inst_pred.extend([0 for i in range(N_ignore)])
    sem_gt.extend([0 for i in range(N_ignore)])
    inst_gt.extend([0 for i in range(N_ignore)])

    # grass segment
    N_grass = n + 1
    N_grass_pred = np.random.randint(0, N_grass)  # rest is sky
    sem_pred.extend([1 for i in range(N_grass_pred)])  # grass
    sem_pred.extend([2 for i in range(N_grass - N_grass_pred)])  # sky
    inst_pred.extend([0 for i in range(N_grass)])
    sem_gt.extend([1 for i in range(N_grass)])  # grass
    inst_gt.extend([0 for i in range(N_grass)])

    # sky segment
    N_sky = n + 2
    N_sky_pred = np.random.randint(0, N_sky)  # rest is grass
    sem_pred.extend([2 for i in range(N_sky_pred)])  # sky
    sem_pred.extend([1 for i in range(N_sky - N_sky_pred)])  # grass
    inst_pred.extend([0 for i in range(N_sky)])  # first instance
    sem_gt.extend([2 for i in range(N_sky)])  # sky
    inst_gt.extend([0 for i in range(N_sky)])  # first instance

    # wrong dog as person prediction
    N_dog = n + 3
    N_person = N_dog
    sem_pred.extend([3 for i in range(N_person)])
    inst_pred.extend([35 for i in range(N_person)])
    sem_gt.extend([4 for i in range(N_dog)])
    inst_gt.extend([22 for i in range(N_dog)])

    # two persons in prediction, but three in gt
    N_person = n + 4
    sem_pred.extend([3 for i in range(6 * N_person)])
    inst_pred.extend([8 for i in range(4 * N_person)])
    inst_pred.extend([95 for i in range(2 * N_person)])
    sem_gt.extend([3 for i in range(6 * N_person)])
    inst_gt.extend([33 for i in range(3 * N_person)])
    inst_gt.extend([42 for i in range(N_person)])
    inst_gt.extend([11 for i in range(2 * N_person)])

    # gt and pred to numpy
    sem_pred = np.array(sem_pred, dtype=np.int64).reshape(1, -1)
    inst_pred = np.array(inst_pred, dtype=np.int64).reshape(1, -1)
    sem_gt = np.array(sem_gt, dtype=np.int64).reshape(1, -1)
    inst_gt = np.array(inst_gt, dtype=np.int64).reshape(1, -1)

    return sem_pred, inst_pred, sem_gt, inst_gt


def test_4D():
    classes = 3  # ignore, car, truck
    cl_strings = ["ignore", "car", "truck"]
    ignore = [0]  # only ignore ignore class

    sem_gt = np.zeros(20, dtype=np.int32)
    sem_gt[5:10] = 1
    sem_gt[10:] = 2

    inst_gt = np.zeros(20, dtype=np.int32)
    inst_gt[5:10] = 1
    inst_gt[10:] = 1
    inst_gt[15:] = 2

    # we have 3 instance 1 car, 2 truck as gt
    sem_pred = np.zeros(20, dtype=np.int32)
    sem_pred[5:10] = 1
    sem_pred[10:15] = 2
    sem_pred[15:] = 1

    inst_pred = np.zeros(20, dtype=np.int32)
    inst_pred[5:10] = 1
    inst_pred[10:] = 2

    # evaluator
    class_evaluator = Panoptic4DEval(3, None, ignore, offset=2**32, min_points=1)
    class_evaluator.addBatch(1, sem_pred, inst_pred, sem_gt, inst_gt)
    PQ4D, AQ_ovr, AQ, iou, iou_mean = class_evaluator.getPQ4D()

    print("=== Results ===")
    print("PQ4D:", PQ4D)
    print("AQ_mean:", AQ_ovr)
    print("AQ:", AQ)
    print("iou:", iou)
    print("iou_mean:", iou_mean)


if __name__ == "__main__":
    # generate problem from He paper (https://arxiv.org/pdf/1801.00868.pdf)
    classes = 5  # ignore, grass, sky, person, dog
    cl_strings = ["ignore", "grass", "sky", "person", "dog"]
    ignore = [0]  # only ignore ignore class
    min_points = 1  # for this example we care about all points

    sem_pred_0, inst_pred_0, sem_gt_0, inst_gt_0 = gen_psuedo_labels(50)
    sem_pred_1, inst_pred_1, sem_gt_1, inst_gt_1 = gen_psuedo_labels(51)

    # evaluator
    evaluator = PanopticEval(classes, ignore=ignore, min_points=1)
    evaluator.addBatch(sem_pred_0, inst_pred_0, sem_gt_0, inst_gt_0)
    evaluator.addBatch(sem_pred_1, inst_pred_1, sem_gt_1, inst_gt_1)

    evaluator_0 = PanopticEval(classes, ignore=ignore, min_points=1)
    evaluator_0.addBatch(sem_pred_0, inst_pred_0, sem_gt_0, inst_gt_0)

    evaluator_1 = PanopticEval(classes, ignore=ignore, min_points=1)
    evaluator_1.addBatch(sem_pred_1, inst_pred_1, sem_gt_1, inst_gt_1)

    pq, sq, rq, all_pq, all_sq, all_rq = evaluator.getPQ()
    iou, all_iou = evaluator.getSemIoU()

    # [PANOPTIC EVAL] IGNORE:  [0]
    # [PANOPTIC EVAL] INCLUDE:  [1 2 3 4]
    # TOTALS
    # PQ: 0.47916666666666663
    # SQ: 0.5520833333333333
    # RQ: 0.6666666666666666
    # IoU: 0.5476190476190476
    # Class ignore 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0
    # Class grass 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
    # Class sky 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
    # Class person 	 PQ: 0.5833333333333333 SQ: 0.875 RQ: 0.6666666666666666 IoU: 0.8571428571428571
    # Class dog 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0

    print("TOTALS")
    print("PQ:", pq.item(), pq.item() == 0.47916666666666663)
    print("SQ:", sq.item(), sq.item() == 0.5520833333333333)
    print("RQ:", rq.item(), rq.item() == 0.6666666666666666)
    print("IoU:", iou.item(), iou.item() == 0.5476190476190476)
    for i, (pq, sq, rq, iou) in enumerate(zip(all_pq, all_sq, all_rq, all_iou)):
        print(
            "Class",
            cl_strings[i],
            "\t",
            "PQ:",
            pq.item(),
            "SQ:",
            sq.item(),
            "RQ:",
            rq.item(),
            "IoU:",
            iou.item(),
        )

    pq, sq, rq, all_pq, all_sq, all_rq = evaluator_0.getPQ()
    iou, all_iou = evaluator_0.getSemIoU()

    # [PANOPTIC EVAL] IGNORE:  [0]
    # [PANOPTIC EVAL] INCLUDE:  [1 2 3 4]
    # TOTALS
    # PQ: 0.47916666666666663
    # SQ: 0.5520833333333333
    # RQ: 0.6666666666666666
    # IoU: 0.5476190476190476
    # Class ignore 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0
    # Class grass 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
    # Class sky 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
    # Class person 	 PQ: 0.5833333333333333 SQ: 0.875 RQ: 0.6666666666666666 IoU: 0.8571428571428571
    # Class dog 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0

    print("TOTALS")
    print("PQ:", pq.item(), pq.item() == 0.47916666666666663)
    print("SQ:", sq.item(), sq.item() == 0.5520833333333333)
    print("RQ:", rq.item(), rq.item() == 0.6666666666666666)
    print("IoU:", iou.item(), iou.item() == 0.5476190476190476)
    for i, (pq, sq, rq, iou) in enumerate(zip(all_pq, all_sq, all_rq, all_iou)):
        print(
            "Class",
            cl_strings[i],
            "\t",
            "PQ:",
            pq.item(),
            "SQ:",
            sq.item(),
            "RQ:",
            rq.item(),
            "IoU:",
            iou.item(),
        )

    pq, sq, rq, all_pq, all_sq, all_rq = evaluator_1.getPQ()
    iou, all_iou = evaluator_1.getSemIoU()

    # [PANOPTIC EVAL] IGNORE:  [0]
    # [PANOPTIC EVAL] INCLUDE:  [1 2 3 4]
    # TOTALS
    # PQ: 0.47916666666666663
    # SQ: 0.5520833333333333
    # RQ: 0.6666666666666666
    # IoU: 0.5476190476190476
    # Class ignore 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0
    # Class grass 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
    # Class sky 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
    # Class person 	 PQ: 0.5833333333333333 SQ: 0.875 RQ: 0.6666666666666666 IoU: 0.8571428571428571
    # Class dog 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0

    print("TOTALS")
    print("PQ:", pq.item(), pq.item() == 0.47916666666666663)
    print("SQ:", sq.item(), sq.item() == 0.5520833333333333)
    print("RQ:", rq.item(), rq.item() == 0.6666666666666666)
    print("IoU:", iou.item(), iou.item() == 0.5476190476190476)
    for i, (pq, sq, rq, iou) in enumerate(zip(all_pq, all_sq, all_rq, all_iou)):
        print(
            "Class",
            cl_strings[i],
            "\t",
            "PQ:",
            pq.item(),
            "SQ:",
            sq.item(),
            "RQ:",
            rq.item(),
            "IoU:",
            iou.item(),
        )

    import pickle

    with open("test.pkl", "wb") as f:
        pickle.dump(evaluator_0, f)

    with open("test.pkl", "rb") as f:
        evaluator_read = pickle.load(f)
    evaluator_1.merge(evaluator_read)

    pq, sq, rq, all_pq, all_sq, all_rq = evaluator_1.getPQ()
    iou, all_iou = evaluator_1.getSemIoU()

    # [PANOPTIC EVAL] IGNORE:  [0]
    # [PANOPTIC EVAL] INCLUDE:  [1 2 3 4]
    # TOTALS
    # PQ: 0.47916666666666663
    # SQ: 0.5520833333333333
    # RQ: 0.6666666666666666
    # IoU: 0.5476190476190476
    # Class ignore 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0
    # Class grass 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
    # Class sky 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
    # Class person 	 PQ: 0.5833333333333333 SQ: 0.875 RQ: 0.6666666666666666 IoU: 0.8571428571428571
    # Class dog 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0

    print("TOTALS")
    print("PQ:", pq.item(), pq.item() == 0.47916666666666663)
    print("SQ:", sq.item(), sq.item() == 0.5520833333333333)
    print("RQ:", rq.item(), rq.item() == 0.6666666666666666)
    print("IoU:", iou.item(), iou.item() == 0.5476190476190476)
    for i, (pq, sq, rq, iou) in enumerate(zip(all_pq, all_sq, all_rq, all_iou)):
        print(
            "Class",
            cl_strings[i],
            "\t",
            "PQ:",
            pq.item(),
            "SQ:",
            sq.item(),
            "RQ:",
            rq.item(),
            "IoU:",
            iou.item(),
        )
