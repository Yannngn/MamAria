import os

import numpy as np
import torch
import wandb
from torchvision.utils import save_image

from calibrators.utils import flatten_logits, softmax_tensor_to_numpy
from utils.post_processing import label_to_pixel

# , get_confidence_of_prediction, fit_ellipses_on_image, get_confidence_of_prediction # noqa: W
from utils.utils import get_device, wandb_mask


def save_predictions_as_imgs(
    data, label, predictions, config, step, dict_eval
):
    img_path = os.path.join(
        config.paths.predictions_dir,
        f"{config.project.time}_pred_e{config.project.epoch}_i{step}.png",
    )

    preds_labels = torch.argmax(predictions, 1)
    preds_img = label_to_pixel(preds_labels, config)

    save_image(preds_img, img_path)

    for j in range(preds_labels.size(0)):
        _id = step * preds_labels.size(0) + j

        local_data = data[j].squeeze(0).cpu().numpy()
        local_label = label[j].cpu().numpy()
        local_pred = preds_labels[j].cpu().numpy()

        wandb_image = wandb_mask(
            local_data, local_label, config.image.labels, local_pred
        )

        dict_eval[f"image_{_id:02d}"] = wandb_image

    wandb.log(dict_eval)


def save_predictions_separated(predictions, path, name, config):
    results = torch.argmax(predictions, 1)
    image = label_to_pixel(results, config)

    for j in range(image.size(0)):
        path = f"{path}/{name}_{j}.png"
        save_image(image[j, :, :, :], path)


def save_calib(model, predictions, config):
    calib_path = os.path.join(
        config.paths.predictions_dir_pos_calib, config.project.time
    )
    os.makedirs(calib_path, exist_ok=True)

    save_predictions_separated(predictions, calib_path, "PRE_calib", config)

    logits = flatten_logits(predictions)
    scores = softmax_tensor_to_numpy(logits)

    results = model.predict_proba(scores)
    results = results.reshape(48, 600, 360, 4)
    results = torch.from_numpy(np.array(results))
    results = results.permute(0, 3, 1, 2)

    save_predictions_separated(results, calib_path, "POS_calib", config)

    return (
        torch.nn.functional.softmax(predictions, 1),
        torch.nn.functional.softmax(results, 1),
    )


def save_validation_as_imgs(loader, config):
    device = get_device(config)
    print("=> Saving validation images ...")

    dict_val = {}

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            img = os.path.join(
                config.paths.predictions_dir,
                f"{config.project.time}_val_i{idx:02d}.png",
            )

            y = y.to(device)
            val = (y / y.max()).unsqueeze(1)

            save_image(val, img)

            for j in range(y.size(0)):
                _id = idx * y.size(0) + j

                local_data = x[j].squeeze(0).cpu().numpy()
                local_label = y[j].cpu().numpy()

                wandb_image = wandb_mask(
                    local_data, local_label, config.image.labels
                )
                dict_val[f"image_{_id:02d}"] = wandb_image

    wandb.log(dict_val)


def save_confidence_as_imgs(predictions, name, config):
    print("=> Saving confidence of prediction as images ...")
    img_path = os.path.join(config.paths.confidences_dir, config.project.time)
    os.makedirs(img_path, exist_ok=True)

    for p, preds in enumerate(predictions):
        for idx, label in enumerate(preds):
            path = os.path.join(img_path, f"confidence_{name}_{p}_{idx}.png")
            label = torch.squeeze(label).type(torch.float32)
            label = (label - torch.min(label)) / (
                torch.max(label) - torch.min(label)
            )  # noqa: W
            save_image(
                label,
                path,
                normalize=False,
            )
