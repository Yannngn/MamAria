from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import label_binarize
from statsmodels.stats.proportion import proportion_confint


def get_binned_scores(labels, scores, bins=10):
    """
    Parameters
    ==========
    labels : array (n_samples, )
        Labels indicating the true class.
    scores : matrix (n_samples, )
        Output probability scores for one or several methods.
    bins : int or list of floats
        Number of bins to create in the scores' space, or list of bin
        boundaries.
    """
    if isinstance(bins, int):
        n_bins = bins
        bins = np.linspace(0, 1 + 1e-8, n_bins + 1)
    elif isinstance(bins, list) or isinstance(bins, np.ndarray):
        n_bins = len(bins) - 1
        bins = np.array(bins)
        if bins[0] == 0.0:
            bins[0] = 0 - 1e-8
        if bins[-1] == 1.0:
            bins[-1] = 1 + 1e-8
    else:
        raise ValueError(f"invalid bins type {type(bins)}")

    scores = np.clip(scores, a_min=0, a_max=1)

    bin_idx = np.digitize(scores, bins) - 1

    bin_true = np.bincount(bin_idx, weights=labels, minlength=n_bins)
    bin_pred = np.bincount(bin_idx, weights=scores, minlength=n_bins)
    bin_total = np.bincount(bin_idx, minlength=n_bins)

    zero_idx = bin_total == 0
    avg_true = np.empty(bin_total.shape[0])
    avg_true.fill(np.nan)
    avg_true[~zero_idx] = np.divide(bin_true[~zero_idx], bin_total[~zero_idx])
    avg_pred = np.empty(bin_total.shape[0])
    avg_pred.fill(np.nan)
    avg_pred[~zero_idx] = np.divide(bin_pred[~zero_idx], bin_total[~zero_idx])
    return avg_true, avg_pred, bin_true, bin_total


def plot_reliability_diagram(
    labels: np.ndarray,
    scores: np.ndarray | list,
    legend: str | None = None,
    show_histogram: bool = True,
    bins: int | List | np.ndarray = 10,
    class_names=None,
    fig=None,
    show_counts: bool = False,
    errorbar_interval=None,
    interval_method: str = "beta",
    fmt: str = "s-",
    show_correction=False,
    show_gaps=False,
    sample_proportion=0,
    hist_per_class=False,
    color_list=None,
    show_bars=False,
    invert_histogram=False,
    color_gaps="lightcoral",
    confidence=False,
):
    """Plots the reliability diagram of the given scores and true labels
    Parameters
    ==========
    labels : array (n_samples, )
        Labels indicating the true class.
    scores : matrix (n_samples, n_classes) or list of matrices
        Output probability scores for one or several methods.
    legend : list of strings or None
        Text to use for the legend.
    show_histogram : boolean
        If True, it generates an additional figure showing the number of
        samples in each bin.
    bins : int or list of floats
        Number of bins to create in the scores' space, or list of bin
        boundaries.
    class_names : list of strings or None
        Name of each class, if None it will assign integer numbers starting
        with 1.
    fig : matplotlib.pyplot.Figure or None
        Figure to use for the plots, if None a new figure is created.
    show_counts : boolean
        If True shows the number of samples of each bin in its corresponding
        line marker.
    errorbar_interval : float or None
        If a float between 0 and 1 is passed, it shows an errorbar
        corresponding to a confidence interval containing the specified
        percentile of the data.
    interval_method : string (default: 'beta')
        Method to estimate the confidence interval which uses the function
        proportion_confint from statsmodels.stats.proportion
    fmt : string (default: 's-')
        Format of the lines following the matplotlib.pyplot.plot standard.
    show_correction : boolean
        If True shows an arrow for each bin indicating the necessary correction
        to the average scores in order to be perfectly calibrated.
    show_gaps : boolean
        If True shows the gap between the average predictions and the true
        proportion of positive samples.
    sample_proportion : float in the interval [0, 1] (default 0)
        If bigger than 0, it shows the labels of the specified proportion of
        samples.
    hist_per_class : boolean
        If True shows one histogram of the bins per class.
    color_list : list of strings or None
        List of string colors indicating the color of each method.
    show_bars : boolean
        If True shows bars instead of lines.
    invert_histogram : boolean
        If True shows the histogram with the zero on top and highest number of
        bin samples at the bottom.
    color_gaps : string
        Color of the gaps (if shown).
    confidence : boolean
        If True shows only the confidence reliability diagram.
    Regurns
    =======
    fig : matplotlib.pyplot.figure
        Figure with the reliability diagram
    """
    if isinstance(scores, list):
        scores_list = scores
    else:
        scores_list = [
            scores,
        ]
    n_scores = len(scores_list)

    if color_list is None:
        color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    classes = np.arange(scores_list[0].shape[1])
    n_classes = len(classes)

    labels = label_binarize(labels, classes=classes)  # type: ignore

    labels_list = []
    if confidence:
        labels_idx = np.argmax(labels, axis=1)
        new_scores_list = []
        for score in scores_list:
            # TODO: randomize selection when there are several winning classes
            conf_idx = np.argmax(score, axis=1)
            winning_score = np.max(score, axis=1)
            new_scores_list.append(np.vstack([1 - winning_score, winning_score]).T)
            labels_list.append((conf_idx.flatten() == labels_idx.flatten()).astype(int))
            labels_list[-1] = label_binarize(labels_list[-1], classes=[0, 1])  # type: ignore
        scores_list = new_scores_list
        n_classes = 2
        class_names = ["Non winning", "winning"]
        n_columns = 1
    else:
        n_columns = labels.shape[1]

    if class_names is None:
        class_names = [str(i + 1) for i in range(n_classes)]

    if n_classes == 2:
        scores_list = [score[:, 1].reshape(-1, 1) for score in scores_list]
        class_names = [
            class_names[1],
        ]

    if fig is None:
        fig = plt.figure(figsize=(n_columns * 4, 4))

    if show_histogram:
        spec = gridspec.GridSpec(
            ncols=n_columns,
            nrows=2,
            height_ratios=[5, 1],
            wspace=0.02,
            hspace=0.04,
            left=0.15,
        )
    else:
        spec = gridspec.GridSpec(ncols=n_columns, nrows=1, hspace=0.04, left=0.15)

    if isinstance(bins, int):
        n_bins = bins
        bins = np.linspace(0, 1 + 1e-8, n_bins + 1)
    elif isinstance(bins, list) or isinstance(bins, np.ndarray):
        n_bins = len(bins) - 1
        bins = np.array(bins)
        if bins[0] == 0.0:
            bins[0] = 0 - 1e-8
        if bins[-1] == 1.0:
            bins[-1] = 1 + 1e-8

    for i in range(n_columns):
        ax1 = fig.add_subplot(spec[i])
        # Perfect calibration
        ax1.plot([0, 1], [0, 1], "--", color="lightgrey", zorder=0)
        for j, score in enumerate(scores_list):
            if labels_list:
                labels = labels_list[j]

            avg_true, avg_pred, bin_true, bin_total = get_binned_scores(labels[:, i], score[:, i], bins=bins)  # type: ignore
            zero_idx = bin_total == 0

            name = legend[j] if legend else None
            if show_bars:
                ax1.bar(
                    x=bins[:-1][~zero_idx],
                    height=avg_true[~zero_idx],
                    align="edge",
                    width=(bins[1:] - bins[:-1])[~zero_idx],
                    edgecolor="black",
                    color=color_list[j],
                )
            else:
                if errorbar_interval is None:
                    ax1.plot(
                        avg_pred,
                        avg_true,
                        fmt,
                        label=name,
                        color=color_list[j],
                    )
                else:
                    nozero_intervals = proportion_confint(
                        count=bin_true[~zero_idx],
                        nobs=bin_total[~zero_idx],
                        alpha=1 - errorbar_interval,
                        method=interval_method,
                    )
                    nozero_intervals = np.array(nozero_intervals)

                    intervals = np.empty((2, bin_total.shape[0]))
                    intervals.fill(np.nan)
                    intervals[:, ~zero_idx] = nozero_intervals

                    yerr = intervals - avg_true
                    yerr = np.abs(yerr)
                    ax1.errorbar(
                        avg_pred,
                        avg_true,
                        yerr=yerr,
                        label=name,
                        fmt=fmt,
                        color=color_list[j],
                    )  # markersize=5)

            if show_counts:
                for ap, at, count in zip(avg_pred, avg_true, bin_total):
                    if np.isfinite(ap) and np.isfinite(at):
                        ax1.text(
                            ap,
                            at,
                            str(count),
                            fontsize=6,
                            ha="center",
                            va="center",
                            zorder=11,
                            bbox=dict(
                                boxstyle="square,pad=0.3",
                                fc="white",
                                ec=color_list[j],
                            ),
                        )

            if show_correction:
                for ap, at in zip(avg_pred, avg_true):
                    ax1.arrow(
                        ap,
                        at,
                        at - ap,
                        0,
                        color=color_gaps,
                        head_width=0.02,
                        length_includes_head=True,
                        width=0.01,
                    )

            if show_gaps:
                for ap, at in zip(avg_pred, avg_true):
                    ygaps = avg_pred - avg_true
                    ygaps = np.vstack((np.zeros_like(ygaps), ygaps))
                    ax1.errorbar(
                        avg_pred,
                        avg_true,
                        yerr=ygaps,
                        fmt=" ",
                        color=color_gaps,
                        lw=4,
                        capsize=5,
                        capthick=1,
                        zorder=10,
                    )

            if sample_proportion > 0:
                idx = np.random.choice(labels.shape[0], int(sample_proportion * labels.shape[0]))
                ax1.scatter(
                    score[idx, i],
                    labels[idx, i],
                    marker="|",  # type: ignore
                    s=100,
                    alpha=0.2,
                    color=color_list[j],
                )

        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.0)
        # ax1.set_title('Class {}'.format(class_names[i]))
        if not show_histogram:
            ax1.set_xlabel("Average score (Class {})".format(class_names[i]))
        if i == 0:
            ax1.set_ylabel("Fraction of positives")
        else:
            ax1.set_yticklabels([])
        ax1.grid(True)
        ax1.set_axisbelow(True)

        if show_histogram:
            ax2 = fig.add_subplot(spec[n_columns + i], label="{}".format(i))
            for j, score in enumerate(scores_list):
                ax1.set_xticklabels([])
                # lines = ax1.get_lines()
                # ax2.set_xticklabels([])

                name = legend[j] if legend else None
                if hist_per_class:
                    for c in [0, 1]:
                        linestyle = ("dotted", "dashed")[c]
                        ax2.hist(
                            score[labels[:, i] == c, i],
                            range=(0, 1),
                            bins=bins,
                            label=name,
                            histtype="step",
                            lw=1,
                            linestyle=linestyle,
                            color=color_list[j],
                            edgecolor="black",
                        )
                else:
                    if n_scores > 1:
                        kwargs = {
                            "histtype": "step",
                            "edgecolor": color_list[j],
                        }
                    else:
                        kwargs = {
                            "histtype": "bar",
                            "edgecolor": "black",
                            "color": color_list[j],
                        }
                    ax2.hist(score[:, i], range=(0, 1), bins=bins, label=name, lw=1, **kwargs)
                ax2.set_xlim(0.0, 1.0)
                ax2.set_xlabel("Average score (Class {})".format(class_names[i]))
                ax2.yaxis.set_major_locator(MaxNLocator(integer=True, prune="upper", nbins=3))
            if i == 0:
                ax2.set_ylabel("Count")
                ytickloc = ax2.get_yticks()
                ax2.yaxis.set_major_locator(mticker.FixedLocator(ytickloc))
                yticklabels = ["{:0.0f}".format(value) for value in ytickloc]
                ax2.set_yticklabels(labels=yticklabels, fontdict=dict(verticalalignment="top"))
            else:
                ax2.set_yticklabels([])
                nbins = len(ax2.get_xticklabels())
                ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune="lower"))
            ax2.grid(True, which="both")
            ax2.set_axisbelow(True)
            if invert_histogram:
                ylim = ax2.get_ylim()
                ax2.set_ylim(reversed(ylim))  # type: ignore

    if legend is not None:
        lines, labels = fig.axes[0].get_legend_handles_labels()  # type: ignore
        fig.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=fig.transFigure,  # type: ignore
            ncol=6,
        )

    fig.align_labels()
    return fig
