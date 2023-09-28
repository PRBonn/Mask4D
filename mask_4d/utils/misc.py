from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def pad_stack(tensor_list: List[Tensor]):
    """
    pad each tensor on the input to the max value in shape[1] and
    concatenate them in a single tensor.
    Input:
        list of tensors [Ni,Pi]
    Output:
        tensor [sum(Ni),max(Pi)]
    """
    _max = max([t.shape[1] for t in tensor_list])
    # valid = [F.pad(torch.ones_like(t), (0, _max - t.shape[1])) for t in tensor_list]
    batched = torch.cat([F.pad(t, (0, _max - t.shape[1])) for t in tensor_list])
    return batched


def apply_pose(points, pose):
    """
    points: [N,3]
    pose: [4,4]
    """
    hpts = torch.hstack((points[:, :3], torch.ones_like(points[:, :1]))).to(points)
    t_pose = torch.tensor(pose).to(points).T
    tr_pts = torch.mm(hpts, t_pose)
    shifted_pts = tr_pts[:, :3]

    return shifted_pts


def center_from_points(points):
    """
    points: instance points Nx3
    return: 3D center [x,y,z]
    """
    xc = (torch.max(points[:, 0]) + torch.min(points[:, 0])) / 2
    yc = (torch.max(points[:, 1]) + torch.min(points[:, 1])) / 2
    zc = (torch.max(points[:, 2]) + torch.min(points[:, 2])) / 2
    return torch.tensor([xc, yc, zc]).to(points)


def fit_ellipse(points):
    """
    fit a 2D ellipse in the XY plane for a set of 3D points
    points: instance points Nx3
    return: axis sizes [size_x, size_y, size_z], angle
    """
    N = points.shape[0]
    X = points[:, :2]
    X = X - X.mean(0)
    _out = torch.linalg.svd(
        (X - X.mean(axis=0)) / torch.sqrt(torch.tensor(N)), full_matrices=False
    )
    angle = np.arctan2(_out.Vh[0, 1].cpu(), _out.Vh[0, 0].cpu()).cuda()
    zc = (torch.max(points[:, 2]) + torch.min(points[:, 2])) / 2
    dz = torch.cdist(points[:, 2].unsqueeze(1), zc.unsqueeze(0).unsqueeze(1))
    sz = 2 * dz.mean()
    size = torch.cat((_out.S, torch.tensor([sz]).to(_out.S)))
    return size, angle


def getcovs(scale_x, scale_y, theta):
    coss = torch.cos(theta)
    sinn = torch.sin(theta)
    rots = torch.stack((torch.stack((coss, -sinn), 1), torch.stack((sinn, coss), 1)), 1)
    covs = torch.stack(
        (
            torch.stack((scale_x, torch.zeros_like(scale_x)), 1),
            torch.stack((torch.zeros_like(scale_y), scale_y), 1),
        ),
        1,
    )
    covs = rots @ covs @ rots.permute(0, 2, 1)
    return covs


def remove_outliers(points):
    m, _ = torch.median(points, 0)
    d = ((points - m) ** 2).sum(1).sqrt()
    return points[d < torch.median(d) * 1.5 * 2]


def bbox_from_pts(points):
    """
    Runs the loss on outputs of the model
    :param points: instance points Nx3
    :return: 3D bbox [x,y,z,theta,l,w,h]
    """

    x1 = torch.min(points[:, 0])
    x2 = torch.max(points[:, 0])
    y1 = torch.min(points[:, 1])
    y2 = torch.max(points[:, 1])
    z1 = torch.min(points[:, 2])
    z2 = torch.max(points[:, 2])

    return torch.tensor(
        [
            x1 + (x2 - x1) / 2,
            y1 + (y2 - y1) / 2,
            z1 + (z2 - z1) / 2,
            0,
            x2 - x1,
            y2 - y1,
            z2 - z1,
        ]
    ).to(points.device)


def th_delete(tensor, indices):
    if len(indices) == 0:
        return tensor
    # remove elements of a tensor at certain indices
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]
