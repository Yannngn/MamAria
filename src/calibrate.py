import logging
import sys

import hydra
import lightning as pl
import pyrootutils
import torch
import wandb
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import DictConfig
from torch.nn import functional as F

from calibrator.minibatch_fulldirichlet import MiniBatchFullDirichletCalibrator

ROOT_DIR = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)

sys.path.append(str(ROOT_DIR))

from src.tools import get_datamodule, get_model_module, setup


@hydra.main(config_path=f"{ROOT_DIR}/config", config_name="testing_penn.yaml", version_base=None)
def calibrate(cfg: DictConfig) -> None:
    cfg = setup(cfg)

    data_module = get_datamodule(cfg, stage="test")

    model_module = get_model_module(cfg)

    profiler = SimpleProfiler()

    trainer = pl.Trainer(**cfg.trainer, profiler=profiler)
    logging.info("Trainer was set. Predicting")

    logging.info("Predict:")

    results = trainer.predict(
        model_module,
        data_module.test_dataloader(),
        ckpt_path=cfg.ckpt_path,
    )

    if results:
        labels = torch.stack([batch[0] for batch in results]).flatten().cpu().numpy()
        logits = torch.stack([batch[1] for batch in results])

        scores = F.softmax(logits, dim=1).permute(0, 2, 3, 1).flatten(end_dim=2).cpu().numpy()

        calibrator = MiniBatchFullDirichletCalibrator(
            num_classes=cfg.num_classes,
            image_shape=cfg.image_shape,
            reg_lambda=1e-2,
            reg_mu=1e-2,
            max_iter=1,
            ref_row=False,
        )

        calibrator.fit(scores, labels)

        logging.info(calibrator.weights)

        # TODO save weights

    logging.info("Done.")


if __name__ == "__main__":
    calibrate()
