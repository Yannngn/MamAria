from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyScore(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, weights: Any | None = None, eps: float = 1e-6) -> None:
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.weights = weights

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.__validate_args(inputs, target)

        softmax_inputs = F.softmax(inputs, dim=1)

        one_hot_target: torch.Tensor = F.one_hot(target, num_classes=inputs.shape[1])
        one_hot_target = one_hot_target.permute(0, 3, 1, 2)

        dims = (0, 2, 3)

        tp = self.weights * torch.sum(softmax_inputs * one_hot_target, dims)
        fp = torch.sum(softmax_inputs * (1 - one_hot_target), dims)
        fn = torch.sum((1 - softmax_inputs) * one_hot_target, dims)

        return torch.divide(tp + self.eps, tp + (self.alpha * fp) + (self.beta * fn) + self.eps)

    def __validate_args(self, inputs: torch.Tensor, target: torch.Tensor):
        assert torch.is_tensor(inputs), f"Input type is not a torch.Tensor. Got {type(inputs)}"
        assert len(inputs.shape) == 4, f"Invalid input shape, we expect BxNxHxW. Got: {inputs.shape}"
        assert (
            inputs.shape[-2:] == target.shape[-2:]
        ), f"input and target shapes must be the same. Got: {inputs.shape} and {target.shape}"
        assert (
            inputs.device == target.device
        ), f"input and target must be in the same device. Got: {inputs.device} and {target.device}"


class TverskyLoss(TverskyScore):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        weights: Any | None = None,
        eps: float = 1e-6,
        reduction: Literal["mean", "sum", "none"] | None = None,
    ) -> None:
        super().__init__(alpha, beta, weights, eps)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky_score = super().forward(inputs, target)
        tversly_loss = 1 - tversky_score

        match self.reduction:
            case "mean":
                return torch.mean(tversly_loss)

            case "sum":
                return torch.sum(tversly_loss)

        return tversly_loss


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
        alpha: float = 0.8,
        gamma: float = 2.0,
        weights: list | None = None,
        eps: float = 1e-6,
        reduction: Literal["mean", "sum"] | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.weights = weights

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.__validate_args(inputs, target)

        one_hot_target: torch.Tensor = F.one_hot(target, num_classes=inputs.shape[1])
        one_hot_target = one_hot_target.permute(0, 3, 1, 2)

        ce_loss = F.cross_entropy(inputs, target, self.weights, reduction="none", label_smoothing=self.eps)
        ce_exp = torch.exp(-ce_loss)

        focal_loss = self.alpha * torch.pow(1 - ce_exp, self.gamma) * ce_loss

        match self.reduction:
            case "mean":
                return torch.mean(focal_loss)

            case "sum":
                return torch.sum(focal_loss)

        return focal_loss

    def __validate_args(self, inputs: torch.Tensor, target: torch.Tensor):
        assert torch.is_tensor(inputs), f"Input type is not a torch.Tensor. Got {type(inputs)}"
        assert len(inputs.shape) == 4, f"Invalid input shape, we expect BxNxHxW. Got: {inputs.shape}"
        assert (
            inputs.shape[-2:] == target.shape[-2:]
        ), f"input and target shapes must be the same. Got: {inputs.shape} and {target.shape}"
        assert (
            inputs.device == target.device
        ), f"input and target must be in the same device. Got: {inputs.device} and {target.device}"


class FocalTverskyLoss(TverskyLoss):
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 1,
        gamma: float = 0.75,
        weights: list | None = None,
        eps: float = 1e-6,
        reduction: Literal["mean", "sum", "none"] | None = None,
    ) -> None:
        super().__init__(alpha, beta, weights, eps)
        self.gamma: float = gamma

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky_loss = super().forward(inputs, target)
        focal_tversky_loss = torch.pow(tversky_loss, self.gamma)

        match self.reduction:
            case "mean":
                return torch.mean(focal_tversky_loss)

            case "sum":
                return torch.sum(focal_tversky_loss)

        return focal_tversky_loss
