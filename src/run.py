import logging
import os
from argparse import ArgumentParser
from getpass import getpass

import lightning as pl
import torch
from omegaconf import OmegaConf

from src.utils.utils import load_obj


def train(cfg: str):
    cfg = OmegaConf.load(cfg)
    logging.basicConfig(
        filename=cfg.general.logging_file,
        encoding="utf-8",
        level=logging.DEBUG,
    )

    try:
        assert os.path.isfile(cfg.data.params.train_csv)
    except AssertionError as err:
        logging.exception(
            """insert a valid path in train_csv at
                          dataset params on the config file."""
        )
        raise err

    pl.seed_everything(cfg.training.params.seed)
    logging.info(f"seed set to {cfg.training.params.seed}")

    data_module = load_obj(cfg.data.class_name)(cfg)
    data_module.setup("training")
    logging.info(f"data module {cfg.data.class_name} was set.")

    model_module = load_obj(cfg.model.class_name)(cfg)
    logging.info(f"model module {cfg.model.class_name} was set.")

    try:
        torch.compile(model_module)
        logging.info("model module was compiled")
    except Exception():
        logging.exception("model module could not be compiled, passing.")
        pass

    setattr(
        cfg.trainer,
        "log_every_n_steps",
        len(data_module.val_dataset) // cfg.data.params.batch_size,
    )

    logger = [
        load_obj(logger["class_name"])(**logger["params"])
        for logger in cfg.logger
    ]

    [
        logging.info(f'logger {logger["class_name"]} was set')
        for logger in cfg.logger
    ]

    callbacks = [
        load_obj(call["class_name"])(**call["params"])
        for call in cfg.callbacks
    ]

    [
        logging.info(f'callback {callback["class_name"]} was set')
        for callback in cfg.callbacks
    ]

    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)

    logging.info("trainer was set. Fitting")

    trainer.fit()


def main():
    train(params.cfg)


if __name__ == "__main__":
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    if not WANDB_API_KEY:
        WANDB_API_KEY = getpass("WANDB API_KEY: ")
        os.environ(WANDB_API_KEY)

    parser = ArgumentParser()
    parser.add_argument("--cfg", type="str", required=True)
    params = parser.parse_args()

    main()
