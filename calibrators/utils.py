import logging
import os

import jax.numpy as jnp
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss

from calibrators.visualization import plot_reliability_diagram
from utils import metrics


def clip_for_log(X, a_min=None, a_max=None):
    """Clip the values in X between a_min and a_max"""
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1 - eps)


def clip(X, a_min=None, a_max=None):
    """Clip the values in X between a_min and a_max"""
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1 - eps)


def clip_jax(X, a_min=None, a_max=None):
    """Clip the values in X between a_min and a_max"""
    eps = jnp.finfo(X.dtype).eps
    return jnp.clip(X, eps, 1 - eps)


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor) -> None:
    ece_criterion = metrics.ECELoss()
    # Torch version
    logits_np = logits.cpu().float().numpy()
    labels_np = labels.cpu().float().numpy()

    # Numpy Version
    logging.info(f"ECE: {ece_criterion.loss(logits_np,labels_np,15):.3f}")

    mce_criterion = metrics.MCELoss()
    logging.info(f"MCE: {mce_criterion.loss(logits_np,labels_np):.3f}")


def plot_reliability(scores, labels, time, odir: bool, calibrated: bool) -> None:
    _ = plot_reliability_diagram(
        labels=labels,
        scores=scores,
        class_names=["Background", "Low", "Mid", "High"],
        show_gaps=True,
        show_bars=True,
        show_histogram=True,
    )

    pre_or_pos = "pos" if calibrated else "pre"
    before_or_after = "After" if calibrated else "Before"

    name = f'{time}_{pre_or_pos}_{"odir" if odir else "full"}.png'
    path = os.path.join("data/plots/", name)

    plt.title(f'{before_or_after} {"Odir" if odir else "Full"} Reliability Diagram')
    plt.savefig(path)


def plot_results(model, scores, labels, time, odir: bool) -> None:
    _ = plot_reliability_diagram(
        labels=labels,
        scores=scores,
        class_names=["Background", "Low", "Mid", "High"],
        show_gaps=True,
        show_bars=True,
        show_histogram=True,
    )

    name = f'{time}_pre_{"odir" if odir else "full"}.png'
    path = os.path.join("data/plots/", name)

    plt.title(f'Before {"Odir" if odir else "Full"} Reliability Diagram')
    plt.savefig(path)

    loss = log_loss(labels, scores)
    logging.info(f"TEST log-loss: UNET {loss:.2f}")

    model.eval()
    with torch.no_grad():
        results = model.predict_proba(scores)

    loss = log_loss(labels, results)
    logging.info(f"TEST log-loss: Calibrator {loss:.2f}")

    _ = plot_reliability_diagram(
        labels=labels,
        scores=results,
        class_names=["Background", "Low", "Mid", "High"],
        show_gaps=True,
        show_bars=True,
        show_histogram=True,
    )

    plt.title(f'After {"Odir" if odir else "Full"} Reliability Diagram')
    name = f'{time}_pos_{"odir" if odir else "full"}.png'
    path = os.path.join("data/plots/", name)
    plt.savefig(path)
