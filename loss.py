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

    def forward(self,input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {type(input))}")
        if not len(input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError(f"input and target shapes must be the same. Got: {input.shape, input.shape} and {target.shape, target.shape}")
        if not input.device == target.device:
            raise ValueError(f"input and target must be in the same device. Got: {input.device, target.device}")
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(input_soft * target_one_hot, dims)
        fps = torch.sum(input_soft * (1. - target_one_hot), dims)
        fns = torch.sum((1. - input_soft) * target_one_hot, dims)

        numerator = self.weights * intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)
        
        return torch.mean(1. - tversky_loss)
