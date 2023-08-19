import logging
import sys

import hydra
import lightning as pl
import pyrootutils
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)

sys.path.append(str(ROOT_DIR))

from src.tools import (
    get_callbacks,
    get_datamodule,
    get_loggers,
    get_model_checkpoint,
    get_model_module,
    get_profiler,
    get_wandb_experiment,
    get_weights,
    setup,
)


@hydra.main(config_path=f"{ROOT_DIR}/config", config_name="v1.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    setup(cfg)
    # TODO save in dir with wandb run name

    data_module = get_datamodule(cfg)
    weights = get_weights(cfg, data_module=data_module)
    logging.info(f"The weights of the dataset are {weights}")

    model_module = get_model_module(cfg, weights)
    loggers = get_loggers(cfg)
    callbacks = get_callbacks(cfg)
    profiler = get_profiler(cfg)

    trainer = pl.Trainer(**cfg.trainer, profiler=profiler, logger=loggers, callbacks=callbacks)
    logging.info("Trainer was set. Fitting")

    trainer.fit(model_module, data_module, ckpt_path=cfg.ckpt_path)
    logging.info("Fit complete.")

    if model_checkpoint := get_model_checkpoint(callbacks):
        best_ckpt = model_checkpoint.best_model_path
        ckpt_score = model_checkpoint.best_model_score

        logging.info(f"Best checkpoint was saved at: {best_ckpt} with score: {ckpt_score:.4f}")
    else:
        logging.info("No ModelCheckpoint to test, exiting.")

        return

    # EVALUATE STEP

    logging.info("Test:")
    trainer.test(model_module, data_module, ckpt_path=best_ckpt)

    # PREDICT STEP

    logging.info("Predict:")
    trainer.predict(model_module, data_module.test_dataloader(), ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
    logging.info("Done.")
