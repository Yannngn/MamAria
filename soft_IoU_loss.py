import torch
import torch.nn as nn
import torch.nn.functional as F

def to_one_hot(tensor, n_classes = 4):
    
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
    
    return one_hot

class SoftIoULoss(nn.Module):
    def __init__(self, weight = None, size_average = True, n_classes=4):
        super(SoftIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_one_hot):
    	# inputs => N x Classes x H x W
    	# target_oneHot => N x Classes x H x W

    	N = inputs.size()[0]

    	# predicted probabilities for each pixel along channel
    	inputs = F.softmax(inputs,dim=1)
    	
    	# Numerator Product
    	inter = inputs * target_one_hot
    	## Sum over all pixels N x C x H x W => N x C
    	inter = inter.view(N,self.classes,-1).sum(2)

    	#Denominator 
    	union= inputs + target_one_hot - (inputs * target_one_hot)
    	## Sum over all pixels N x C x H x W => N x C
    	union = union.view(N,self.classes,-1).sum(2)

    	loss = inter/union

    	## Return average loss over classes and batch
    	return -loss.mean()