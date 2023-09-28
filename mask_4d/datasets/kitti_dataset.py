import os
import random

import numpy as np
import torch
import yaml
from mask_4d.utils.data_util import data_prepare
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SemanticDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.things_ids = []
        self.color_map = []

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_set = SemanticDataset(
            self.cfg.KITTI.PATH + "/sequences/",
            self.cfg.KITTI.CONFIG,
            split="train",
        )
        train_mask_set = MaskSemanticDataset(
            dataset=train_set,
            split="train",
            min_pts=self.cfg.KITTI.MIN_POINTS,
            space=self.cfg.KITTI.SPACE,
            num_pts=self.cfg.KITTI.SUB_NUM_POINTS,
            voxel_size=self.cfg.BACKBONE.VOXEL_SIZE,
            voxel_max=self.cfg.BACKBONE.VOXEL_MAX,
        )
        self.train_seq_mask = SequenceMaskDataset(
            train_mask_set,
            n_scans=self.cfg.TRAIN.N_SCANS,
            interval=self.cfg.TRAIN.INTERVAL,
        )

        val_set = SemanticDataset(
            self.cfg.KITTI.PATH + "/sequences/",
            self.cfg.KITTI.CONFIG,
            split="valid",
        )
        self.val_mask_set = MaskSemanticDataset(
            dataset=val_set,
            split="valid",
            min_pts=self.cfg.KITTI.MIN_POINTS,
            space=self.cfg.KITTI.SPACE,
            num_pts=0,
            voxel_size=self.cfg.BACKBONE.VOXEL_SIZE,
            voxel_max=self.cfg.BACKBONE.VOXEL_MAX,
        )

        test_set = SemanticDataset(
            self.cfg.KITTI.PATH + "/sequences/",
            self.cfg.KITTI.CONFIG,
            split="test",
        )
        self.test_mask_set = MaskSemanticDataset(
            dataset=test_set,
            split="test",
            min_pts=self.cfg.KITTI.MIN_POINTS,
            space=self.cfg.KITTI.SPACE,
            num_pts=0,
            voxel_size=self.cfg.BACKBONE.VOXEL_SIZE,
            voxel_max=self.cfg.BACKBONE.VOXEL_MAX,
        )

        self.things_ids = train_set.things_ids
        self.color_map = train_set.color_map

    def train_dataloader(self):
        dataset = self.train_seq_mask
        collate_fn = SphericalSequenceCollation()
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        dataset = self.val_mask_set
        collate_fn = SphericalBatchCollation()
        self.valid_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)
        return self.valid_loader

    def test_dataloader(self):
        dataset = self.test_mask_set
        collate_fn = SphericalBatchCollation()
        self.test_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.test_iter = iter(self.test_loader)
        return self.test_loader


class SemanticDataset(Dataset):
    def __init__(self, data_path, cfg_path, split="train"):
        yaml_path = cfg_path
        with open(yaml_path, "r") as stream:
            semyaml = yaml.safe_load(stream)

        self.things = get_things()
        self.stuff = get_stuff()

        self.things_ids = get_things_ids()

        self.color_map = semyaml["color_map_learning"]
        self.labels = semyaml["labels"]
        self.learning_map = semyaml["learning_map"]
        self.inv_learning_map = semyaml["learning_map_inv"]
        self.split = split
        split = semyaml["split"][self.split]

        self.im_idx = []
        pose_files = []
        calib_files = []
        fill = 2
        for i_folder in split:
            self.im_idx += absoluteFilePaths(
                "/".join([data_path, str(i_folder).zfill(fill), "velodyne"])
            )
            pose_files.append(
                absoluteDirPath(
                    "/".join([data_path, str(i_folder).zfill(fill), "poses.txt"])
                )
            )
            calib_files.append(
                absoluteDirPath(
                    "/".join([data_path, str(i_folder).zfill(fill), "calib.txt"])
                )
            )

        self.im_idx.sort()
        self.poses, self.n_scans = load_poses(pose_files, calib_files)

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        fname = self.im_idx[index]
        pose = self.poses[index]
        points = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        xyz = points[:, :3]
        intensity = points[:, 3]
        if len(intensity.shape) == 2:
            intensity = np.squeeze(intensity)
        if self.split == "test":
            annotated_data = np.expand_dims(
                np.zeros_like(points[:, 0], dtype=int), axis=1
            )
            sem_labels = annotated_data
            ins_labels = annotated_data
        else:
            annotated_data = np.fromfile(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
                dtype=np.int32,
            ).reshape((-1, 1))
            sem_labels = annotated_data & 0xFFFF
            ins_labels = annotated_data >> 16
            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)

        return (xyz, sem_labels, ins_labels, intensity, fname, pose)


class MaskSemanticDataset(Dataset):
    def __init__(
        self,
        dataset,
        split,
        min_pts,
        space,
        num_pts,
        voxel_size,
        voxel_max,
    ):
        self.dataset = dataset
        self.num_points = num_pts
        self.split = split
        self.min_points = min_pts
        self.th_ids = dataset.things_ids
        self.xlim = space[0]
        self.ylim = space[1]
        self.zlim = space[2]
        self.voxel_size = np.array(voxel_size)
        self.voxel_max = voxel_max

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        xyz, sem_labels, ins_labels, intensity, fname, pose = data
        full_xyz = xyz.shape[0]
        keep_xyz = np.argwhere(
            (self.xlim[0] < xyz[:, 0])
            & (xyz[:, 0] < self.xlim[1])
            & (self.ylim[0] < xyz[:, 1])
            & (xyz[:, 1] < self.ylim[1])
            & (self.zlim[0] < xyz[:, 2])
            & (xyz[:, 2] < self.zlim[1])
        )[:, 0]
        xyz = xyz[keep_xyz]
        sem_labels = sem_labels[keep_xyz]
        ins_labels = ins_labels[keep_xyz]
        intensity = intensity[keep_xyz]

        feats = np.concatenate((xyz, np.expand_dims(intensity, axis=1)), axis=1)

        # Subsample
        if self.split == "train" and len(xyz) > self.num_points:
            idx = np.random.choice(np.arange(len(xyz)), self.num_points, replace=False)
            xyz = xyz[idx]
            sem_labels = sem_labels[idx]
            ins_labels = ins_labels[idx]
            feats = feats[idx]
            intensity = intensity[idx]

        # SphereFormer
        sp_xyz = xyz.copy()
        sp_coords, sp_xyz, sp_feats, sp_labels, sp_idx_recons = data_prepare(
            sp_xyz,
            feats,
            sem_labels,
            self.split,
            self.voxel_size,
            self.voxel_max,
        )

        if self.split == "test":
            return (
                xyz,
                sem_labels,
                ins_labels,
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                fname,
                pose,
                keep_xyz,
                full_xyz,
                sp_coords,
                sp_xyz,
                sp_feats,
                sp_labels,
                sp_idx_recons,
            )

        stuff_masks = np.array([]).reshape(0, xyz.shape[0])
        stuff_masks_ids = []
        things_masks = np.array([]).reshape(0, xyz.shape[0])
        things_cls = np.array([], dtype=int)
        things_masks_ids = []

        stuff_labels = np.asarray(
            [0 if s in self.th_ids else s for s in sem_labels[:, 0]]
        )
        stuff_cls, st_cnt = np.unique(stuff_labels, return_counts=True)
        # filter small masks
        keep_st = np.argwhere(st_cnt > self.min_points)[:, 0]
        stuff_cls = stuff_cls[keep_st][1:]
        if len(stuff_cls):
            stuff_masks = np.array(
                [np.where(stuff_labels == i, 1.0, 0.0) for i in stuff_cls]
            )
            stuff_masks_ids = [0 for m in stuff_masks]
        # things masks
        ins_sems = np.where(ins_labels == 0, 0, sem_labels)
        _ins_labels = ins_sems + ((ins_labels << 16) & 0xFFFF0000).reshape(-1, 1)
        things_ids, th_idx, th_cnt = np.unique(
            _ins_labels[:, 0], return_index=True, return_counts=True
        )
        # filter small instances
        keep_th = np.argwhere(th_cnt > self.min_points)[:, 0]
        things_ids = things_ids[keep_th]
        th_idx = th_idx[keep_th]
        # remove instances with wrong sem class
        keep_th = np.array(
            [i for i, idx in enumerate(th_idx) if sem_labels[idx] in self.th_ids],
            dtype=int,
        )
        things_ids = things_ids[keep_th]
        th_idx = th_idx[keep_th]
        if len(th_idx):
            things_masks = np.array(
                [np.where(_ins_labels[:, 0] == i, 1.0, 0.0) for i in things_ids]
            )
            things_cls = np.array([sem_labels[i] for i in th_idx]).squeeze(1)
            things_masks_ids = [t for t in things_ids]

        masks = torch.from_numpy(np.concatenate((stuff_masks, things_masks)))
        masks_cls = torch.from_numpy(np.concatenate((stuff_cls, things_cls)))
        stuff_masks_ids.extend(things_masks_ids)
        masks_ids = torch.tensor(stuff_masks_ids)

        assert (
            masks.shape[0] == masks_cls.shape[0]
        ), f"not same number masks and classes: masks {masks.shape[0]}, classes {masks_cls.shape[0]} "

        return (
            xyz,
            sem_labels,
            ins_labels,
            masks,
            masks_cls,
            masks_ids,
            fname,
            pose,
            keep_xyz,
            full_xyz,
            sp_coords,
            sp_xyz,
            sp_feats,
            sp_labels,
            sp_idx_recons,
        )


class SequenceMaskDataset(Dataset):
    def __init__(self, dataset, n_scans, interval):
        super().__init__()
        self.dataset = dataset
        self.n_scans = n_scans
        self.interval = interval

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        fname = data[6]
        _fname = fname.split("/")
        scan = int(_fname[-1][:-4])
        seq = _fname[-3]
        max_scan = self.dataset.dataset.n_scans[seq] - 1

        # select n_scans randomly between scan-interval/2 and scan+interval/2
        s_before = random.sample(
            set(np.arange(scan - int(self.interval / 2), scan)), round(self.n_scans / 2)
        )
        s_after = random.sample(
            set(np.arange(scan, scan + int(self.interval / 2))), self.n_scans // 2
        )
        scans = s_before + s_after
        scans.sort(reverse=True)
        while scans[-1] < 0:
            scans.pop()
        scans.sort()
        while scans[-1] > max_scan:
            scans.pop()
        idx = index + np.array(scans) - scan
        np.append(idx, np.array([index]))
        idx.sort()
        out_data = []
        for i in idx:
            out_data.append(self.dataset[i])

        return out_data


class SphericalBatchCollation:
    def __init__(self):
        self.keys = [
            "pt_coord",
            "sem_label",
            "ins_label",
            "masks",
            "masks_cls",
            "masks_ids",
            "fname",
            "pose",
            "keep_xyz",
            "full_xyz",
            "sp_coord",
            "sp_xyz",
            "sp_feat",
            "sp_label",
            "sp_idx_recons",
        ]

    def __call__(self, data):
        return {self.keys[i]: list(x) for i, x in enumerate(zip(*data))}


class SphericalSequenceCollation:
    def __init__(self):
        self.keys = [
            "pt_coord",
            "sem_label",
            "ins_label",
            "masks",
            "masks_cls",
            "masks_ids",
            "fname",
            "pose",
            "keep_xyz",
            "full_xyz",
            "sp_coord",
            "sp_xyz",
            "sp_feat",
            "sp_label",
            "sp_idx_recons",
        ]

    def __call__(self, data):
        _data = data[0]  # always bs=0
        return {self.keys[i]: list(x) for i, x in enumerate(zip(*_data))}


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def absoluteDirPath(directory):
    return os.path.abspath(directory)


def parse_calibration(filename):
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        calib[key] = pose
    calib_file.close()
    return calib


def parse_poses(filename, calibration):
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses


def load_poses(pose_files, calib_files):
    poses = []
    n_scans = {}
    # go through every file and get all poses
    # add them to match im_idx
    for i in range(len(pose_files)):
        calib = parse_calibration(calib_files[i])
        seq_poses_f64 = parse_poses(pose_files[i], calib)
        seq_poses = [pose.astype(np.float32) for pose in seq_poses_f64]
        poses += seq_poses
        seq = pose_files[i].split("/")[-2]
        n_scans[seq] = len(seq_poses)
    return poses, n_scans


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


def get_things():
    things = {
        1: "car",
        2: "bicycle",
        3: "motorcycle",
        4: "truck",
        5: "other-vehicle",
        6: "person",
        7: "bicyclist",
        8: "motorcyclist",
    }
    return things


def get_stuff():
    stuff = {
        9: "road",
        10: "parking",
        11: "sidewalk",
        12: "other-ground",
        13: "building",
        14: "fence",
        15: "vegetation",
        16: "trunk",
        17: "terrain",
        18: "pole",
        19: "traffic-sign",
    }
    return stuff


def get_things_ids():
    return [1, 2, 3, 4, 5, 6, 7, 8]
