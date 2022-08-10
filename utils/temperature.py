
import torch
from torch import nn, optim
from torch.nn import functional as F

from utils import metrics

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        #print(f'{logits.shape=}')
        temperature = self.temperature.unsqueeze(1).expand(logits.shape)
        #print(f'{temperature.shape=}')
        return torch.div(logits, temperature)

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        #self.cuda()
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = metrics.ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                label = label.long()
                input = input
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
                # logits_list.append(logits)
                # labels_list.append(label)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion.loss(logits.numpy(),labels.numpy(),15)
        #before_temperature_ece = ece_criterion(logits, labels).item()
        #ece_2 = ece_criterion_2.loss(logits,labels)
        print(f'Before temperature - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')
        #print(ece_2)
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion.loss(self.temperature_scale(logits).detach().numpy(),labels.numpy(), 15)
        print(f'Optimal temperature: {self.temperature.item():.3f}')
        print(f'After temperature - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')

        return self