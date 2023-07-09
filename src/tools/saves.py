import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.utils as TU
from torch import Tensor


def torch_save_prediction(
    image: Tensor,
    prediction: dict[str, Tensor],
    image_name: str,
    output_dir: str | Path,
    colors: Any | None = None,
) -> None:
    """Receives a detection prediction which is a dict containing masks,
    boxes and scores, writes the information in the corresponding original
    image and saves the result in the correct path

    Args:
        prediction (Dict[str, Tensor]): one prediction
        image_path (str): path of the corresponding image
        target (str): target name = {'farol', 'lanterna', ...}
        result_dir (str): location of the results
        colors (Optional[List], optional): Custom colors for each mask
        predicted. Defaults to None.
        gray_value (int, optional): Custom value for the mask- and box-only
        image. Defaults to 255.
    """
    car_id = image_name.split(".")[0]
    image = (image * 255).to(torch.uint8)
    black = torch.zeros_like(image)

    masks = prediction["masks"].squeeze().bool()
    boxes = prediction["boxes"]

    masked: Tensor = TU.draw_segmentation_masks(image, masks, colors=colors)
    only_masks: Tensor = TU.draw_segmentation_masks(black, masks, alpha=0, colors=colors)
    boxed: Tensor = TU.draw_bounding_boxes(image, boxes, colors=colors)
    only_boxes: Tensor = TU.draw_bounding_boxes(black, boxes, colors=colors, fill=True)

    path = os.path.join(output_dir, car_id)

    os.makedirs(path, exist_ok=True)

    TU.save_image(masked.to(torch.float32) / 255, os.path.join(path, f"masked.png"))
    TU.save_image(boxed.to(torch.float32) / 255, os.path.join(path, f"boxed.png"))
    TU.save_image(only_masks.to(torch.float32) / 255, os.path.join(path, f"masks.png"))
    TU.save_image(only_boxes.to(torch.float32) / 255, os.path.join(path, f"boxes.png"))


def torch_save_predictions(
    images: list[Tensor],
    predictions: list[dict[str, Tensor]],
    image_names: list[Any],
    output_dir: str | Path,
    colors: Any | None = None,
) -> None:
    """
    Receives all batches and save all predictions in the path specified on the
    config file

    Args:
        predictions (List[List[Dict[str, Tensor]]]): List of batches, each
        batch is a list of dicts each one containing a prediction, keys
        {'masks', 'boxes', 'scores'}
        image_paths (List[str]): list of image paths
        cfg (Union[ListConfig, DictConfig]): config object made by OmegaConf
        colors (Optional[List], optional): Custom colors for each mask
        predicted. Defaults to None.
        gray_value (int, optional): Custom value for the mask- and box-only
        image. Defaults to 255.
    """
    for image_name, image, prediction in zip(image_names, images, predictions):
        torch_save_prediction(
            image=image,
            prediction=prediction,
            image_name=str(image_name),
            output_dir=output_dir,
            colors=colors,
        )


def cv2_save_mask(idx: int, car_id: str, mask: np.ndarray, params: Namespace) -> None:
    path = os.path.join(getattr(params, "output_path"), car_id)

    os.makedirs(path, exist_ok=True)
    target = getattr(params, "target")
    path = os.path.join(f"{target}_{idx}.png")

    cv2.imwrite(path, mask)


def cv2_save_image(car_id: str, image: np.ndarray, params: Namespace, name: str = "masked") -> None:
    path = os.path.join(getattr(params, "output_path"), car_id)

    os.makedirs(path, exist_ok=True)
    target = getattr(params, "target")
    path = os.path.join(path, f"{name}_{target}.png")

    cv2.imwrite(path, image)


def save_torch_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    params: Namespace,
    epoch: int | str = 0,
    time: str = "0",
    final: bool = False,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "params": vars(params),
    }

    try:
        model_name = os.path.basename(getattr(params, "checkpoint_path")).split(".")[0]
    except TypeError:
        model_name = f"{time}_{params.target}"

    output_path = getattr(params, "checkpoint_output_path", f"{params.target}/checkpoints/")
    checkpoint_name = f'{model_name}_{"final" if final else epoch}.pickle'
    save_path = os.path.join(output_path, checkpoint_name)

    os.makedirs(output_path, exist_ok=True)
    torch.save(state, save_path)
