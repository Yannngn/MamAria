import json
import torch

from datetime import datetime
from munch import munchify
from yaml import safe_load

from utils.save_images import save_predictions_as_imgs, save_submission_as_imgs, save_ellipse_pred_as_imgs, save_confidence_as_imgs
from utils.utils import check_accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('config.yaml') as f:
    CONFIG = munchify(safe_load(f))

def log_early_stop(val_loader, model, loss_train, loss_val, epoch, time=0, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=DEVICE):
    log_predictions(val_loader, model, loss_train, loss_val, epoch, time, folder, device)

def log_predictions(val_loader, model, loss_train, loss_val, epoch, time=0, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=DEVICE):
    num_correct, num_pixels, dict_eval  = check_accuracy(val_loader, model, device)

    dict_eval['epoch'] = epoch
    dict_eval['loss_train'] = loss_train
    dict_eval['loss_val'] = loss_val

    print(f"Got {num_correct} of {num_pixels} pixels;")

    json_lines = json.dumps(dict_eval)
    
    with open(folder+f'{time}_prediction.json','w') as f:
        json.dump(json_lines, f)  
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_predictions_as_imgs(val_loader, model, epoch, dict_eval, time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)
    #save_ellipse_pred_as_imgs(val_loader, model, epoch, dict_eval, time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)
    save_confidence_as_imgs(val_loader, model, epoch, dict_eval, time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)

def log_submission(loader, model, loss_test, time=0, folder=CONFIG.PATHS.SUBMISSIONS_DIR, device = DEVICE):
    num_correct, num_pixels, dict_subm = check_accuracy(loader, model, device)
    dict_subm['loss_subm'] = loss_test

    print(f"Got {num_correct} of {num_pixels} pixels;")
    # for key in dict_subm:
    #     print (key,':', dict_subm[key])
    
    #json_lines = json.dumps(dict_subm)
    
    with open(folder+f'{time}_submission.json','w') as f:
        json.dump(dict_subm, f, ensure_ascii=False, indent=4)  

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    #save_submission_as_imgs(loader, model, dict_subm, time=now, folder=CONFIG.PATHS.SUBMISSIONS_DIR, device=device)
    save_confidence_as_imgs(loader, model, 0, dict(), time=now, folder=CONFIG.PATHS.PREDICTIONS_DIR, device=device)