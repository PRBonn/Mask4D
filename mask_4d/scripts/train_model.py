import os
import subprocess
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from mask_4d.datasets.kitti_dataset import SemanticDatasetModule
from mask_4d.models.mask_model import Mask4D
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@click.command()
@click.option("--w", type=str, default=None, required=False)
@click.option("--ckpt", type=str, default=None, required=False)
def main(w, ckpt):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/decoder.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})

    data = SemanticDatasetModule(cfg)
    model = Mask4D(cfg)
    if w:
        w = torch.load(w, map_location="cpu")
        model.load_state_dict(w["state_dict"])

    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID, default_hp_metric=False
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    aq_ckpt = ModelCheckpoint(
        monitor="metrics/aq",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_aq{metrics/aq:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )
    pq_ckpt = ModelCheckpoint(
        monitor="metrics/pq",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_pq{metrics/pq:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )
    pq4d_ckpt = ModelCheckpoint(
        monitor="metrics/pq4d",
        filename=cfg.EXPERIMENT.ID + "_epoch{epoch:02d}_pq4d{metrics/pq4d:.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
    )

    callbacks = [lr_monitor, pq_ckpt, pq4d_ckpt, aq_ckpt]

    trainer = Trainer(
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="ddp",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=callbacks,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=ckpt,
    )

    trainer.fit(model, data)


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


if __name__ == "__main__":
    main()
