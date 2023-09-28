import os

import numpy as np


def save_results(sem_preds, ins_preds, output_dir, batch, class_inv_lut):
    for i in range(len(sem_preds)):
        sem = sem_preds[i]
        ins = ins_preds[i]
        sem_inv = class_inv_lut[sem].astype(np.uint32)
        label = sem_inv.reshape(-1, 1) + (
            (ins.astype(np.uint32) << 16) & 0xFFFF0000
        ).reshape(-1, 1)

        pcd_path = batch["fname"][i]
        seq = pcd_path.split("/")[-3]
        pcd_fname = pcd_path.split("/")[-1].split(".")[-2] + ".label"
        fname = os.path.join(output_dir, seq, "predictions", pcd_fname)
        label.reshape(-1).astype(np.uint32).tofile(fname)


def full_preds(x, _sem_pred, _ins_pred):
    sem_pred = []
    ins_pred = []
    for i in range(len(x["full_xyz"])):
        sem = np.zeros(x["full_xyz"][i]).astype(np.int64) * 9
        sem[x["keep_xyz"][i]] = _sem_pred[i]
        sem_pred.append(sem)
        ins = np.zeros(x["full_xyz"][i]).astype(np.int64) * 9
        ins[x["keep_xyz"][i]] = _ins_pred[i]
        ins_pred.append(ins)
    return sem_pred, ins_pred
