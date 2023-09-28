# Modified by Rodrigo Marcuzzi from https://github.com/dvlab-research/SphereFormer/blob/master/util/data_util.py
import numpy as np
import torch
from mask_4d.utils.voxelize import voxelize
from torch_scatter import scatter_mean


def data_prepare(
    coord,
    feat,
    label,
    split="train",
    voxel_size=np.array([0.1, 0.1, 0.1]),
    voxel_max=None,
):
    coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord_norm = coord - coord_min
    if split == "train":
        uniq_idx, idx_recon = voxelize(coord_norm, voxel_size)
        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(voxel_size))
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        if voxel_max and label.shape[0] > voxel_max:
            init_idx = np.random.randint(label.shape[0])
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[
                :voxel_max
            ]
            coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
            coord_voxel = coord_voxel[crop_idx]
    else:
        idx_recon = voxelize(coord_norm, voxel_size, mode=1)

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    if split == "train":
        coord_voxel = torch.LongTensor(coord_voxel)
        idx_recon = torch.LongTensor(idx_recon)
        return coord_voxel, coord, feat, label, idx_recon
    else:
        coord_norm = torch.FloatTensor(coord_norm)
        idx_recon = torch.LongTensor(idx_recon)
        coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
        coord_voxel = torch.floor(coord_norm / torch.from_numpy(voxel_size)).long()
        coord = scatter_mean(coord, idx_recon, dim=0)
        feat = scatter_mean(feat, idx_recon, dim=0)
        return coord_voxel, coord, feat, label, idx_recon
