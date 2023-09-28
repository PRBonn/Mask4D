import time

import numpy as np
import yaml
from mask_4d.utils.eval_np import Panoptic4DEval


class PanopticKitti4DEvaluator:
    def __init__(self, cfg):
        dataset_config_file = cfg.CONFIG
        self.load_kitti_config(dataset_config_file)
        min_points = 50
        self.evaluator = Panoptic4DEval(
            self.nr_classes,
            None,
            self.ignore_class,
            offset=2**32,
            min_points=min_points,
        )
        self.class_metrics = {}
        self.mean_metrics = {}

    def reset(self):
        self.evaluator.reset()
        self.class_metrics = {}
        self.mean_metrics = {}

    def load_kitti_config(self, config_file):
        # Load semantic-kitti config
        # data = yaml.safe_load(open('datasets/semantic-kitti.yaml', 'r'))
        data = yaml.safe_load(open(config_file, "r"))
        # get number of interest classes, and the label mappings
        class_remap = data["learning_map"]
        self.class_inv_remap = data["learning_map_inv"]
        class_ignore = data["learning_ignore"]
        self.nr_classes = len(self.class_inv_remap)
        self.class_strings = data["labels"]
        # make lookup table for mapping
        maxkey = max(class_remap.keys())
        # +100 hack making lut bigger just in case there are unknown labels
        class_lut = np.zeros((maxkey + 100), dtype=np.int32)
        class_lut[list(class_remap.keys())] = list(class_remap.values())
        self.ignore_class = [cl for cl, ignored in class_ignore.items() if ignored]

        self.class_inv_lut = np.zeros((20), dtype=np.int32)
        self.class_inv_lut[list(self.class_inv_remap.keys())] = list(
            self.class_inv_remap.values()
        )

        self.things = get_things()
        self.stuff = get_stuff()
        self.all_classes = self.things + self.stuff

    def update(self, sem_preds, ins_preds, inputs):
        for i in range(len(sem_preds)):
            self.evaluator.addBatch(
                inputs["fname"][i].split("/")[-3],
                sem_preds[i],
                ins_preds[i],
                inputs["sem_label"][i].reshape(-1),
                inputs["ins_label"][i].reshape(-1),
            )

    def get_class_inv_lut(self):
        return self.class_inv_lut

    def calculate_metrics(self):
        PQ4D, AQ_mean, AQ, iou, iou_mean = self.evaluator.getPQ4D()
        things_iou = iou[1:9].mean()
        stuff_iou = iou[9:].mean()

        # print("=== Results ===")
        # print("PQ4D:", PQ4D)
        # print("AQ_mean:", AQ_mean)
        # print("AQ:", AQ)
        # print("iou:", iou)
        # print("iou_mean:", iou_mean)
        # print("things_iou:", things_iou)
        # print("stuff_iou:", stuff_iou)

        metrics_dict = {}
        metrics_dict["PQ4D"] = float(PQ4D)
        metrics_dict["AQ_mean"] = float(AQ_mean)
        metrics_dict["iou_mean"] = float(iou_mean)
        metrics_dict["things_iou"] = things_iou
        metrics_dict["stuff_iou"] = stuff_iou

        # Save mean metrics
        self.mean_metrics = metrics_dict

        # Save to a file
        # with open(output_filename, 'w') as outfile:
        # yaml.dump(codalab_output, outfile, default_flow_style=False)

        class_all_IoU = iou.flatten().tolist()
        class_all_AQ = AQ.flatten().tolist()

        classes_dict = {}
        for idx, (aq, iou) in enumerate(zip(class_all_AQ, class_all_IoU)):
            class_str = self.class_strings[self.class_inv_remap[idx]]
            classes_dict[class_str] = {}
            classes_dict[class_str]["AQ"] = aq
            classes_dict[class_str]["IoU"] = iou

        # Save per class metrics
        self.class_metrics = classes_dict

    def get_mean_pq4d(self):
        return self.mean_metrics["PQ4D"]

    def get_mean_aq(self):
        return self.mean_metrics["AQ_mean"]

    def get_mean_iou(self):
        return self.mean_metrics["iou_mean"]

    def get_class_metrics(self):
        return self.class_metrics

    def print_results(self):
        print("|        |   AQ   |   IoU  |")
        for k, v in self.class_metrics.items():
            print(
                "|{}| {:.4f} | {:.4f} |".format(
                    k.ljust(8)[-8:],
                    v["AQ"],
                    v["IoU"],
                )
            )
        for key in self.mean_metrics.keys():
            print("{}:\t{}".format(key, self.mean_metrics[key]))

    def print_fp_fn(self):
        print("True Positive: ")
        print("\t|\t".join([str(x) for x in self.evaluator.pan_tp]))
        print("False Positive: ")
        print("\t|\t".join([str(x) for x in self.evaluator.pan_fp]))
        print("False Negative: ")
        print("\t|\t".join([str(x) for x in self.evaluator.pan_fn]))


def get_things():
    things = [
        "car",
        "bicycle",
        "motorcycle",
        "truck",
        "other-vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
    ]
    return things


def get_stuff():
    stuff = [
        "road",
        "parking",
        "sidewalk",
        "other-ground",
        "building",
        "fence",
        "vegetation",
        "trunk",
        "terrain",
        "pole",
        "traffic-sign",
    ]
    return stuff
