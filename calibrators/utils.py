import logging
import os

import jax.numpy as jnp
import numpy as np
import torch
from matplotlib import pyplot as plt
from pycalib.visualisations import plot_reliability_diagram
from sklearn.metrics import log_loss

from utils import metrics


def clip_for_log(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1 - eps)


def clip(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1 - eps)


def clip_jax(X):
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
