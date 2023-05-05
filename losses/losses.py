import logging
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float,
        gamma: float | None = 2.0,
        weights: List | None = None,
        reduction: Literal["none", "mean", "sum"] | None = None,
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: Literal["none", "mean", "sum"] | None = reduction
        self.eps: float = 1e-6
        self.weights: List | None = weights

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
        softmax_inputs = F.softmax(inputs, dim=1) + self.eps

        # create the labels one hot tensor
        one_hot_target: torch.Tensor = F.one_hot(target, num_classes=inputs.shape[1])
        one_hot_target = one_hot_target.permute(0, 3, 1, 2)

        # compute the actual focal loss
        dims = (0, 2, 3)
        weight = torch.pow(1.0 - softmax_inputs, self.gamma)
        focal = -self.alpha * weight * torch.log(softmax_inputs)

        loss_tmp = self.weights * torch.mean(one_hot_target * focal, dim=dims)

        if self.reduction in ["none", None]:
            return loss_tmp
        elif self.reduction == "mean":
            return torch.mean(loss_tmp)
        elif self.reduction == "sum":
            return torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, weights: List | None = None) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = 1e-6
        self.weights: List | None = weights

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
        softmax_inputs = F.softmax(inputs, dim=1)

        # create the labels one hot tensor
        one_hot_target: torch.Tensor = F.one_hot(target, num_classes=inputs.shape[1])
        one_hot_target = one_hot_target.permute(0, 3, 1, 2)

        # compute the dice score
        dims = (0, 2, 3)
        intersection = self.weights * torch.sum(softmax_inputs * one_hot_target, dims)

        fp = torch.sum(softmax_inputs * (1 - one_hot_target), dims)
        fn = torch.sum((1 - softmax_inputs) * one_hot_target, dims)

        denom = intersection + (self.alpha * fp) + (self.beta * fn)

        tversky_score = intersection / (denom + self.eps)

        return torch.mean(1.0 - tversky_score)
