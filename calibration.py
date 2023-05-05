import logging
import os
from typing import List, Tuple

import matplotlib as mpl
import numpy as np
import torch
from memory_profiler import profile
from munch import munchify
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from yaml import safe_load

from calibrators.fulldirichlet import FullDirichletCalibrator
from calibrators.minibatch_fulldirichlet import MiniBatchFullDirichletCalibrator
from calibrators.utils import calibration_metrics, plot_reliability, plot_results
from models.unet import UNET
from utils.utils import (
    get_device,
    get_loaders,
    get_loss_function,
    get_time,
    get_transforms,
    load_checkpoint,
)


def load_torch_model(config):
    device = get_device(config)
    assert os.path.isfile(config.load.path), "checkpoint path was not provided"

    model = UNET(config).to(device)
    model = nn.DataParallel(model)

    load_checkpoint(torch.load(config.load.path), model, None, None)

    return model


def load_data(config):
    _, val_transforms, calib_transforms = get_transforms(config)
    _, val_loader, calib_loader = get_loaders(config, None, val_transforms, calib_transforms)

    return val_loader, calib_loader


def main(config):
    prediction_path = "data/predictions/"
    odir = True
    lambda_ = [1e-2]  # [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    mu_ = lambda_ if odir else [None]

    model = load_torch_model(config)
    loss_fn = get_loss_function(config)
    val_loader, calib_loader = load_data(config)

    if os.path.isfile(os.path.join(prediction_path, "val_scores.npy")):
        scores = np.load(os.path.join(prediction_path, "val_scores.npy"))
        labels = np.load(os.path.join(prediction_path, "val_labels.npy"))
    else:
        scores, labels = predict(model, val_loader, loss_fn)
        np.save(os.path.join(prediction_path, "val_scores.npy"), scores)
        np.save(os.path.join(prediction_path, "val_labels.npy"), labels)

    plot_reliability(scores, labels, NOW, odir, False)

    print(scores.min(), scores.max())
    calibrator = calibrate(scores, labels, lambda_, mu_)
    logging.info(calibrator.weights)

    del scores, labels

    if os.path.isfile(os.path.join(prediction_path, "calib_scores.npy")):
        scores = np.load(os.path.join(prediction_path, "calib_scores.npy"))
        labels = np.load(os.path.join(prediction_path, "calib_labels.npy"))
    else:
        scores, labels = predict(model, calib_loader, loss_fn)
        np.save(os.path.join(prediction_path, "calib_scores.npy"), scores)
        np.save(os.path.join(prediction_path, "calib_labels.npy"), labels)

    plot_reliability(scores, labels, NOW, odir, True)


def calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    lambda_: List[float],
    mu_: List[float | None],
):
    mini_batch = True
    if mini_batch:
        calibrator = MiniBatchFullDirichletCalibrator(reg_lambda=lambda_[0], reg_mu=mu_[0], max_iter=1, ref_row=False)
    else:
        calibrator = FullDirichletCalibrator(reg_lambda=lambda_[0], reg_mu=mu_[0])

    calibrator.fit(scores, labels)
    return calibrator

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

    gscv = GridSearchCV(
        calibrator,
        cv=skf,
        scoring="neg_log_loss",
        param_grid={"reg_lambda": lambda_, "reg_mu": mu_},
        n_jobs=1,
        verbose=1,
    )

    logging.info(f"logits.shape: {scores.shape}, labels.shape: {labels.shape}")

    gscv.fit(
        scores,
        labels,
    )

    logging.info("Grid of parameters cross-validated")
    logging.info(gscv.param_grid)
    logging.info(f"Best parameters: {gscv.best_params_}")

    return gscv.best_estimator_


def predict(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = get_device(config)
    loop = tqdm(loader)
    logits_list, labels_list = [], []
    correct, total = 0, 0

    model.eval()
    for images, labels in loop:
        images, labels = images.to(device), labels.long().to(device)

        # forward
        with torch.no_grad():
            logits = model(images)
            loss = loss_fn(logits, labels)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # append to list
        logits_list.append(logits)
        labels_list.append(labels)

        # convert to probabilities
        output_probs = F.softmax(logits, dim=1)

        # pick max args
        _, predicted = torch.max(output_probs, 1)

        # total
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    loop.close()

    pixels = config.image.image_height * config.image.image_width * total

    logging.info(
        f"""Accuracy of the network on the {pixels}
        test pixels: {100 * correct / pixels:.3f} %"""
    )

    logits, labels = torch.cat(logits_list), torch.cat(labels_list)
    calibration_metrics(logits, labels)

    scores = F.softmax(logits, dim=1)
    scores = scores.permute(0, 2, 3, 1).flatten(end_dim=2).cpu().numpy()
    labels = labels.flatten().cpu().numpy()

    return scores, labels


def save_example(
    model,
    loader,
    loss_fn,
    width: int = 360,
    height: int = 600,
    output_path: str = "data/predictions/",
    num: int | None = None,
    prefix: str | None = None,
):
    scores, labels = predict(model, loader, loss_fn)

    if not prefix:
        prefix = ""

    prefix = f"{num}_{prefix}" if num else prefix

    np.save(
        os.path.join(output_path, f"{prefix}images_scores.npy"),
        scores[: num * width * height],
    )
    np.save(
        os.path.join(output_path, f"{prefix}images_labels.npy"),
        labels[: num * width * height],
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("jax._src.lib.xla_bridge").addFilter(lambda _: False)
    # warnings.filterwarnings("ignore")

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    NOW = get_time()

    mpl.use("Agg")

    with open("config_prediction.yaml") as f:
        config = munchify(safe_load(f))

    main(config)
