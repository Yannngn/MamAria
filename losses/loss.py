import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, weight: list) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = 1e-6
        self.weight: list = weight

    def forward(
        self, inputs: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_tensor(inputs):
            raise TypeError(
                f"""Input type is not a torch.Tensor. Got {type(inputs)}"""
            )
        if not len(inputs.shape) == 4:
            raise ValueError(
                f"""Invalid input shape, we expect BxNxHxW.
                Got: {inputs.shape}"""
            )
        if not inputs.shape[-2:] == target.shape[-2:]:
            raise ValueError(
                f"""input and target shapes must be the same.
                Got: {inputs.shape} and {target.shape}"""
            )
        if not inputs.device == target.device:
            raise ValueError(
                f"""input and target must be in the same device.
                Got: {inputs.device, target.device}"""
            )
        # compute softmax over the classes axis
        num_classes = inputs.shape[1]
        if num_classes == 1:
            target_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
            target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
            target_1_hot_f = target_1_hot[:, 0:1, :, :]
            target_1_hot_s = target_1_hot[:, 1:2, :, :]
            target_1_hot = torch.cat([target_1_hot_s, target_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(inputs)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            target_1_hot = torch.eye(num_classes)[target.squeeze(1)]
            target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(inputs, dim=1)
        target_1_hot = target_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(self.weight * probas * target_1_hot, dims)
        fps = torch.sum(probas * (1 - target_1_hot), dims)
        fns = torch.sum((1 - probas) * target_1_hot, dims)
        num = intersection
        denom = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_loss = (num / (denom + self.eps)).mean()

        return torch.mean(1.0 - tversky_loss)
