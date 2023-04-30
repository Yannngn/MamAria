from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from utils import metrics


class ConfidenceHistogram(metrics.MaxProbCELoss):
    def plot(
        self,
        output: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15,
        logits: bool = True,
        title: Optional[str] = None,
    ) -> plt.Axes:
        super().loss(output, labels, n_bins, logits)
        # scale each datapoint
        n = len(labels)
        w = np.ones(n) / n

        plt.rcParams["font.family"] = "serif"
        # size and axis limits
        plt.figure(figsize=(3, 3))
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        ticks = np.arange(0, 1.2, 0.2).tolist()
        plt.xticks(ticks, map(str, ticks))
        plt.yticks(ticks, map(str, ticks))

        # plot grid
        plt.grid(
            color="tab:grey", linestyle=(0, (1, 5)), linewidth=1, zorder=0
        )
        # plot histogram
        plt.hist(
            self.confidences,
            n_bins,
            weights=w,
            color="b",
            range=(0.0, 1.0),
            edgecolor="k",
        )

        # plot vertical dashed lines
        acc = np.mean(self.accuracies)
        conf = np.mean(self.confidences)
        plt.axvline(x=acc, color="tab:grey", linestyle="--", linewidth=3)
        plt.axvline(x=conf, color="tab:grey", linestyle="--", linewidth=3)
        if acc > conf:
            plt.text(acc + 0.03, 0.9, "Accuracy", rotation=90, fontsize=11)
            plt.text(
                conf - 0.07, 0.9, "Avg. Confidence", rotation=90, fontsize=11
            )
        else:
            plt.text(acc - 0.07, 0.9, "Accuracy", rotation=90, fontsize=11)
            plt.text(
                conf + 0.03, 0.9, "Avg. Confidence", rotation=90, fontsize=11
            )

        plt.ylabel("% of Samples", fontsize=13)
        plt.xlabel("Confidence", fontsize=13)
        plt.tight_layout()

        if title is not None:
            plt.title(title, fontsize=16)

        return plt


class ReliabilityDiagram(metrics.MaxProbCELoss):
    def plot(
        self,
        output: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15,
        logits: bool = True,
        title: Optional[str] = None,
    ) -> plt.Axes:
        super().loss(output, labels, n_bins, logits)

        # computations
        delta = 1.0 / n_bins
        x = np.arange(0, 1, delta)
        mid = np.linspace(delta / 2, 1 - delta / 2, n_bins)
        error = np.abs(np.subtract(mid, self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        # size and axis limits
        plt.figure(figsize=(3, 3))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # plot grid
        plt.grid(
            color="tab:grey", linestyle=(0, (1, 5)), linewidth=1, zorder=0
        )
        # plot bars and identity line
        plt.bar(
            x,
            self.bin_acc,
            color="b",
            width=delta,
            align="edge",
            edgecolor="k",
            label="Outputs",
            zorder=5,
        )
        plt.bar(
            x,
            error,
            bottom=np.minimum(self.bin_acc, mid),
            color="mistyrose",
            alpha=0.5,
            width=delta,
            align="edge",
            edgecolor="r",
            hatch="/",
            label="Gap",
            zorder=10,
        )

        ident = [0.0, 1.0]
        plt.plot(ident, ident, linestyle="--", color="tab:grey", zorder=15)

        # labels and legend
        plt.ylabel("Accuracy", fontsize=13)
        plt.xlabel("Confidence", fontsize=13)
        plt.legend(loc="upper left", framealpha=1.0, fontsize="medium")
        if title is not None:
            plt.title(title, fontsize=16)
        plt.tight_layout()

        return plt
