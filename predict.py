import logging
import os
import torch
import wandb
import warnings

from datetime import datetime
from munch import munchify, unmunchify
from tqdm import tqdm
from yaml import safe_load

from loggers.logs import log_predictions
from models.unet import UNET
from utils.utils import get_loaders, load_checkpoint, get_device, get_metrics, get_transforms, get_loss_function

warnings.filterwarnings("ignore")

def predict_fn(test_loader, model, loss_fn, global_metrics, label_metrics, config):
    device = get_device(config)
    loop = tqdm(test_loader)

    model.eval()

    for idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.long().to(device)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        log_predictions(data, targets, predictions, global_metrics, label_metrics, config, idx)

    return loss.item()

def main(config):
    device = get_device(config)
    logging.info('predict')
    wandb.init(project=config.wandb.project_name,
               entity=config.wandb.project_team,
               config=unmunchify(config.hyperparameters))

    config.hyperparameters = munchify(wandb.config)

    _, _, test_transforms = get_transforms(config)
    _, _, test_loader = get_loaders(config, None, None, test_transforms)

    model = UNET(config).to(device)
    model = torch.nn.DataParallel(model)

    load_checkpoint(torch.load(config.load_model.path), 
                    model, 
                    optimizer=None, 
                    scheduler=None
                    )

    '''
    model = UNET(config)
    load_checkpoint(torch.load(CHECKPOINT, 
                               map_location=torch.device('cpu')), 
                    model, 
                    optimizer=None, 
                    scheduler=None
                    )

    model = model.to(device)
    model = torch.nn.DataParallel(model)
    
    '''

    loss_fn = get_loss_function(config)

    global_metrics, label_metrics = get_metrics(config)
    
    predict_fn(test_loader,
               model,
               loss_fn,
               global_metrics, 
               label_metrics,
               config,
               )
    
    wandb.finish()  

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    #CHECKPOINT = "data/checkpoints/20220619_061736_best_checkpoint.pth.tar"

    with open('config_prediction.yaml') as f:
        config = munchify(safe_load(f))  

    os.environ['WANDB_MODE'] = 'online' if config.wandb.online else 'offline'
    config.project.time = datetime.now().strftime("%Y%m%d_%H%M%S")

    main(config)