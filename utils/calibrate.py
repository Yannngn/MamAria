import logging
from typing import List, Optional

import torch
from matplotlib import pyplot as plt
from pycalib.visualisations import plot_reliability_diagram
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

from utils import metrics


def flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.permute(0, 2, 3, 1)
    return logits.flatten(end_dim=2)


def softmax_tensor_to_numpy(logits: torch.Tensor):
    scores = torch.nn.functional.softmax(logits, dim=1)
    return scores.cpu().float().numpy()


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor) -> None:
    ece_criterion = metrics.ECELoss()
    # Torch version
    logits_np = logits.cpu().float().numpy()
    labels_np = labels.cpu().float().numpy()

    # Numpy Version
    logging.log(f"ECE: {ece_criterion.loss(logits_np,labels_np,15):.3f}")

    mce_criterion = metrics.MCELoss()
    logging.log(f"MCE: {mce_criterion.loss(logits_np,labels_np):.3f}")


def fit_calibrator(
    calibrator,
    logits: torch.Tensor,
    labels: torch.Tensor,
    lambda_: list,
    mu_: Optional[List],
    sample: Optional[List] = None,
):
    logits = flatten_logits(logits)
    scores = softmax_tensor_to_numpy(logits)

    labels = labels.flatten().cpu().float().numpy()

    gscv = GridSearchCV(
        calibrator,
        scoring="neg_log_loss",
        param_grid={"reg_lambda": lambda_, "reg_mu": mu_ if mu_ else [None]},
        n_jobs=1,
        verbose=1,
    )

    if sample:
        gscv.fit(scores[sample], labels[sample])
    else:
        gscv.fit(scores, labels)

    logging.log("Grid of parameters cross-validated")
    logging.log(gscv.param_grid)
    logging.log(f"Best parameters: {gscv.best_params_}")

    return gscv


def plot_results(model, scores, labels, time, odir: bool) -> None:
    _ = plot_reliability_diagram(
        labels=labels,
        scores=scores,
        class_names=["Background", "Low", "Mid", "High"],
        show_gaps=True,
        show_bars=True,
        show_histogram=True,
    )

    plt.title(f'Before {"Odir" if odir else "Full"} Reliability Diagram')
    plt.savefig(f'data/plots/{time}_pre_{"odir" if odir else "full"}.png')

    loss = log_loss(labels, scores)
    logging.log(f"TEST log-loss: UNET {loss:.2f}")

    model.eval()
    with torch.no_grad():
        results = model.predict_proba(scores)

    loss = log_loss(labels, results)
    logging.log(f"TEST log-loss: Calibrator {loss:.2f}")

    _ = plot_reliability_diagram(
        labels=labels,
        scores=results,
        class_names=["Background", "Low", "Mid", "High"],
        show_gaps=True,
        show_bars=True,
        show_histogram=True,
    )

    plt.title(f'After {"Odir" if odir else "Full"} Reliability Diagram')
    plt.savefig(f'data/plots/{time}_pos_{"odir" if odir else "full"}.png')
