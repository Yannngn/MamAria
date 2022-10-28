import logging
import matplotlib as mpl
import torch

from datetime import datetime
from munch import munchify
from tqdm import tqdm
from typing import Tuple
from yaml import safe_load

from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
                                     
from models.unet import UNET

from utils.calibrate import calibration_metrics, fit_calibrator, flatten_logits, softmax_tensor_to_numpy, plot_results
from utils.fulldirichletcustom import FullDirichletCalibratorCustom
from utils.utils import get_device, get_loaders, get_loss_function, get_transforms, load_checkpoint

def main(config):
    device = get_device(config)
    assert config.load_model.path, print(f'checkpoint path was not provided')
    
    _, val_transforms, calib_transforms = get_transforms(config) 
    _, val_loader, calib_loader = get_loaders(config, None, val_transforms, calib_transforms)

    model = UNET(config).to(device)
    model = torch.nn.DataParallel(model) 
    
    load_checkpoint(torch.load(config.load.path), model, None, None)
    
    loss_fn = get_loss_function(config)

    logits, labels = predict(model, val_loader, loss_fn, device)

    calibration_metrics(logits, labels)

    if config.calibration.custom:
        calibrator = FullDirichletCalibratorCustom(reg_lambda=lambda_, reg_mu=mu_)
    else:
        calibrator = FullDirichletCalibrator(reg_lambda=lambda_, reg_mu=mu_)
   
    odir = False
    lambda_ = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    mu_ = lambda_ if odir else None

    # sample = random.sample(range(0, torch.numel(labels)), 1 * torch.numel(labels) // (1 * 4))
    # gscv = fit_calibrator(logits, labels, lambda_, mu_, sample)

    gscv = fit_calibrator(calibrator, logits, labels, lambda_, mu_)
    
    logits, labels = predict(model, calib_loader, loss_fn, device)
    logits = flatten_logits(logits)
    scores = softmax_tensor_to_numpy(logits)
    labels = labels.flatten().cpu().numpy()
    
    plot_results(gscv, scores, labels, NOW, odir)

def predict(model, loader, loss_fn, device) -> Tuple[torch.Tensor, torch.Tensor]:
    loop = tqdm(loader)
    logits_list, labels_list = [], []
    correct, total = 0, 0

    model.eval()
    for images, labels in loop:
        images, labels = images.to(device), labels.long().to(device)
        
        # forward
        with torch.no_grad():
            logits = model(images)
            loss = loss_fn(logits, labels)
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        # append to list
        logits_list.append(logits)
        labels_list.append(labels)
        
        # convert to probabilities
        output_probs = torch.nn.functional.softmax(logits, dim=1)
        
        # pick max args
        _, predicted = torch.max(output_probs, 1)
        
        # total
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    loop.close()

    pixels = config.image.image_height * config.image.image_width * total
    logging.info(f'Accuracy of the network on the {pixels} test pixels: {100 * correct / pixels:.3f} %')

    return torch.cat(logits_list), torch.cat(labels_list)   

if __name__ == '__main__':
    NOW = datetime.now().strftime("%m%d%Y-%H%M%S")
    CUSTOM = True
    
    mpl.use('Agg')
    
    with open('config_prediction.yaml') as f:
        config = munchify(safe_load(f))  

    main(config)
    


    



