import os
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from mask_4d.datasets.kitti_dataset import SemanticDatasetModule
from mask_4d.models.mask_model import Mask4D
from pytorch_lightning import Trainer, seed_everything


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option("--w", type=str, required=True)
@click.option("--save_testset", is_flag=True)
def main(w, save_testset):
    seed_everything(42, workers=True)
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

    cfg.EVALUATE = True
    if save_testset:
        results_dir = create_dirs()
        print(f"Saving test set predictions in directory {results_dir}")
        cfg.RESULTS_DIR = results_dir

    data = SemanticDatasetModule(cfg)
    model = Mask4D(cfg)
    w = torch.load(w, map_location="cpu")
    model.load_state_dict(w["state_dict"])

    trainer = Trainer(gpus=cfg.TRAIN.N_GPUS)

    if save_testset:
        trainer.test(model, data)
    else:
        trainer.validate(model, data)
    model.evaluator.print_results()
    print("#############################################################")
    model.evaluator4d.print_results()


def create_dirs():
    results_dir = join(getDir(__file__), "..", "output", "test", "sequences")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    for i in range(11, 22):
        sub_dir = os.path.join(results_dir, str(i).zfill(2), "predictions")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
    return results_dir


if __name__ == "__main__":
    main()
