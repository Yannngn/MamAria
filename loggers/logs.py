from utils.metrics import check_accuracy
from utils.save_images import (
    save_calib,
    save_confidence_as_imgs,
    save_predictions_as_imgs,
)


def log_predictions(
    data, label, predictions, global_metrics, label_metrics, config, step
):
    dict_eval = check_accuracy(
        predictions, label, global_metrics, label_metrics
    )

    save_predictions_as_imgs(data, label, predictions, config, step, dict_eval)


def log_calib(
    label,
    predictions,
    global_metrics,
    label_metrics,
    config,
    loader,
    model,
    device,
):
    pre, pos = save_calib(model, predictions, config)

    dict_pre_calib = check_accuracy(pre, label, global_metrics, label_metrics)
    dict_pos_calib = check_accuracy(pos, label, global_metrics, label_metrics)

    save_confidence_as_imgs(pre, "pre", config)
    save_confidence_as_imgs(pos, "pos", config)

    with open("dict_pre_calib.csv", "w") as f:
        for key in dict_pre_calib.keys():
            f.write("%s,%s\n" % (key, dict_pre_calib[key]))

    with open("dict_pos_calib.csv", "w") as f:
        for key in dict_pos_calib.keys():
            f.write("%s,%s\n" % (key, dict_pos_calib[key]))

    return dict_pos_calib, dict_pre_calib
