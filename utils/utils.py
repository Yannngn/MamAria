import logging
import os
from datetime import datetime
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import torch
import torchmetrics
from albumentations.augmentations.geometric import resize
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import wandb
from losses import losses


def get_time():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_device(config):
    if config.project.device == "gpu":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # noqa: W

    return torch.device("cpu")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    logging.info("Saving checkpoint ...")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, scheduler):
    logging.info("Loading checkpoint...")

    try:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        return
    except KeyError:
        pass

    try:
        model.load_state_dict(checkpoint)
    except KeyError as e:
        raise ValueError(f'Key {e} is different from expected "state_dict"')

    return checkpoint.project.epoch


def get_loaders(config, train_transform, val_transform, test_transform):
    if config.image.phantom_format == "dcm":
        from datasets.dataset import PhantomDCMDataset as PhantomDataset
    elif config.image.phantom_format == "tiff":
        from datasets.dataset import PhantomTIFFDataset as PhantomDataset

    train_dir = config.paths.train_img_dir
    train_maskdir = config.paths.train_mask_dir
    val_dir = config.paths.val_img_dir
    val_maskdir = config.paths.val_mask_dir
    test_dir = config.paths.test_img_dir
    test_maskdir = config.paths.test_mask_dir

    num_workers = min([4, config.project.num_workers])
    pin_memory = config.project.pin_memory

    batch_size = config.hyperparameters.batch_size

    if train_transform:
        train_ds = PhantomDataset(
            image_dir=train_dir,
            mask_dir=train_maskdir,
            transform=train_transform,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
    else:
        train_loader = None

    if val_transform:
        val_ds = PhantomDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    else:
        val_loader = None

    if test_transform:
        test_ds = PhantomDataset(image_dir=test_dir, mask_dir=test_maskdir, transform=test_transform)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    else:
        val_loader = None

    return train_loader, val_loader, test_loader


def get_weights(config):
    device = get_device(config)
    weights = np.zeros(config.image.mask_labels)
    multiplier = np.array(config.hyperparameters.multiplier)
    mask_dir = config.paths.train_mask_dir
    total_pixels = 0
    mask_files = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith(".png")]

    for mask in mask_files:
        temp = []
        mask = cv2.imread(mask)

        if total_pixels == 0:
            total_pixels = mask.shape[1] * mask.shape[2]

        for i in range(config.image.mask_labels):
            temp.append((mask == i).sum())

        weights += temp

    den = weights / (total_pixels * len(mask))
    out = np.divide(
        multiplier,
        den,
        out=np.zeros_like(multiplier, dtype=float),
        where=den != 0,
    )

    return torch.tensor(out).float().to(device)


def get_loss_function(config, loss_fn_name: Literal["crossentropy", "tversky", "focal"] | None = None):
    if not loss_fn_name:
        loss_fn_name = config.hyperparameters.loss_fn

    try:
        assert loss_fn_name in [None, "crossentropy", "tversky", "focal"]
    except AssertionError as aerr:
        logging.error(f"{loss_fn_name} is not a recognized loss function")
        raise aerr

    weights = get_weights(config) if config.hyperparameters.weights else None  # noqa: W

    if loss_fn_name == "crossentropy":
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_fn_name == "tversky":
        return losses.TverskyLoss(
            alpha=config.hyperparameters.tversky_alpha, beta=config.hyperparameters.tversky_beta, weights=weights
        )
    elif loss_fn_name == "focal":
        return losses.FocalLoss(
            alpha=config.hyperparameters.focal_alpha,
            gamma=config.hyperparameters.focal_gamma,
            weights=weights,
            reduction="sum",
        )


def get_optimizer(config, parameters, optim_list=["adam", "sgd"]):
    optimizer = config.hyperparameters.optimizer_fn

    try:
        assert optimizer in optim_list
    except AssertionError as aerr:
        logging.error(f"{optimizer} is not a recognized optimizer")
        raise aerr

    if optimizer == "adam":
        return optim.Adam(parameters, lr=config.hyperparameters.adam_learning_rate)
    elif optimizer == "sgd":
        return optim.SGD(
            parameters,
            lr=config.hyperparameters.sgd_learning_rate,
            momentum=config.hyperparameters.sgd_momentum,
            weight_decay=config.hyperparameters.sgd_weight_decay,
            nesterov=config.hyperparameters.sgd_nesterov,
        )


def get_scheduler(config, optimizer, scheduler_list=["plateau", "cosine", "cyclic", "warm"]):
    scheduler_fn = config.hyperparameters.scheduler_fn

    try:
        assert scheduler_fn in scheduler_list
    except AssertionError as aerr:
        logging.error(
            f"""{scheduler_fn} is not a recognized
                      lr scheduler function"""
        )
        raise aerr

    if scheduler_fn == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=config.hyperparameters.scheduler_patience,
        )
    elif scheduler_fn == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer)
    elif scheduler_fn == "cyclic":
        return lr_scheduler.CyclicLR(optimizer, max_lr=0.001, base_lr=0.00001)
    elif scheduler_fn == "warm":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.hyperparameters.scheduler_patience)
    else:
        raise KeyError(f"scheduler {scheduler_fn} not recognized.")


def get_metrics(config):
    num_classes = config.image.mask_labels
    average = "weighted" if config.hyperparameters.weights else "micro"

    device = get_device(config)

    global_metrics = [
        torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, mdmc_average="global").to(device),
        torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        ).to(device),
        torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        ).to(device),
        torchmetrics.Precision(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        ).to(device),
        torchmetrics.Recall(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        ).to(device),
        torchmetrics.Specificity(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        ).to(device),
        torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        ).to(device),
    ]

    global_metrics_names = [
        "global accuracy",
        "weighted accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "jaccard",
    ]

    label_metrics = [
        torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device),
        torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average=None).to(device),
        torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=None).to(device),
        torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=None).to(device),
        torchmetrics.Specificity(task="multiclass", num_classes=num_classes, average=None).to(device),
        torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, average=None).to(device),
    ]

    label_metrics_names = [
        "accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "jaccard",
    ]

    global_dict = dict(zip(global_metrics_names, global_metrics))
    label_dict = dict(zip(label_metrics_names, label_metrics))

    return global_dict, label_dict


def get_transforms(config):
    fmt = config.image.phantom_format
    assert fmt == "dcm" or fmt == "tiff", f"image format {fmt} is not accepted"

    if fmt == "dcm":
        height, width = config.image.image_height, config.image.image_width
        train_transforms = A.Compose(
            [
                # 3584, 2816 -> 3000, 1800
                A.augmentations.crops.transforms.Crop(1016, 292, 2816, 3292),
                resize.Resize(height, width, interpolation=cv2.INTER_LANCZOS4),
                ToTensorV2(),
            ]
        )

        val_transforms = A.Compose(
            [
                A.augmentations.crops.transforms.Crop(1016, 292, 2816, 3292),
                resize.Resize(height, width, interpolation=cv2.INTER_LANCZOS4),
                ToTensorV2(),
            ]
        )

        test_transforms = A.Compose(
            [
                A.augmentations.crops.transforms.Crop(1016, 292, 2816, 3292),
                resize.Resize(height, width, interpolation=cv2.INTER_LANCZOS4),
                ToTensorV2(),
            ],
        )

    elif fmt == "tiff":
        train_transforms = A.Compose(
            [
                ToTensorV2(),
            ],
        )
        val_transforms = A.Compose(
            [
                ToTensorV2(),
            ],
        )
        test_transforms = A.Compose(
            [
                ToTensorV2(),
            ],
        )

    return train_transforms, val_transforms, test_transforms


def wandb_mask(data, true_mask, labels, prediction=None):
    if prediction is not None:
        return wandb.Image(
            data,
            masks={
                "prediction": {
                    "mask_data": prediction,
                    "class_labels": labels,
                },
                "ground truth": {
                    "mask_data": true_mask,
                    "class_labels": labels,
                },
            },
        )

    return wandb.Image(
        data,
        masks={"ground truth": {"mask_data": true_mask, "class_labels": labels}},
    )
