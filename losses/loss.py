import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, weights: list) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = 1e-6
        self.weights: list = weights

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert torch.is_tensor(inputs), f"Input type is not a torch.Tensor. Got {type(inputs)}"
        assert len(inputs.shape) == 4, f"Invalid input shape, we expect BxNxHxW. Got: {inputs.shape}"
        assert (
            inputs.shape[-2:] == target.shape[-2:]
        ), f"input and target shapes must be the same. Got: {inputs.shape} and {target.shape}"
        assert (
            inputs.device == target.device
        ), f"input and target must be in the same device. Got: {inputs.device} and {target.device}"

        # compute softmax over the classes axis
        logging.info(f"{inputs.shape=}")
        softmax_inputs = F.softmax(inputs, dim=1)

        # create the labels one hot tensor
        logging.info(f"{target.shape=}")
        one_hot_target = F.one_hot(target, num_classes=inputs.shape[1])
        logging.info(f"{one_hot_target.shape=}")
        # compute the dice score
        dims = (1, 2, 3)

        intersection = torch.sum(self.weights * softmax_inputs * one_hot_target, dims)

        logging.info(f"{intersection.shape=}")

        fp = torch.sum(softmax_inputs * (1 - one_hot_target), dims)
        fn = torch.sum((1 - softmax_inputs) * one_hot_target, dims)

        denom = intersection + (self.alpha * fp) + (self.beta * fn)

        tversky_score = intersection / (denom + self.eps)

        return torch.mean(1.0 - tversky_score)
