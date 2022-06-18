import torch
import wandb

from torchvision.utils import save_image

from utils.post_processing import label_to_pixel#, fit_ellipses_on_image, get_confidence_of_prediction
from utils.utils import get_device, wandb_mask

def save_predictions_as_imgs(data, label, predictions, config, step, dict_eval):
    img_path = config.paths.predictions_dir + f"{config.time}_pred_e{config.epoch}_i{step}.png"
    
    #if step == 0: print("=> Saving predictions as images ...")
        
    preds_labels = torch.argmax(predictions, 1)
    preds_img = label_to_pixel(preds_labels, config)
    
    save_image(preds_img, img_path)
    
    for j in range(preds_labels.size(0)):
        _id = step * preds_labels.size(0) + j
        
        local_data = data[j].squeeze(0).cpu().numpy()
        local_label = label[j].cpu().numpy()
        local_pred = preds_labels[j].cpu().numpy()
        
        wandb_image = wandb_mask(local_data, local_label, config.image.labels, local_pred)
        
        dict_eval[f'image_{_id:02d}'] = wandb_image

    wandb.log(dict_eval)

'''def save_submission_as_imgs(loader, model, dict_subm, folder=CONFIG.PATHS.SUBMISSIONS_DIR, time=0, device=DEVICE):
    print("=> Saving submission images ...")
    
    model.eval()
    
    with torch.no_grad():    
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            preds_labels = torch.argmax(model(x), 1)
            preds = (preds_labels / preds_labels.max()).unsqueeze(1)
            preds_img = label_to_pixel(preds_labels)

            for j in range(preds_labels.size(0)):
                local_data = x[j].squeeze(0).cpu().numpy()
                local_label = y[j].cpu().numpy()
                local_pred = preds[j].cpu().numpy()               
                img = folder + f"{time}_submission_i{idx:02d}_p{j:02d}.png"
                save_image(preds_img[j], img)

                wandb_image = wandb_mask(local_data, local_label, CONFIG.IMAGE.LABELS, local_pred)
                dict_subm[f'submission_i{idx:02d}_p{j:02d}'] = wandb_image
            
                wandb.log(dict_subm)'''

def save_validation_as_imgs(loader, config):
    device = get_device(config)
    print("=> Saving validation images ...")
    
    dict_val = {}
    
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            img = f"{config.paths.predictions_dir}{config.time}_val_i{idx:02d}.png"
            
            y = y.to(device)
            val = (y / y.max()).unsqueeze(1)
            
            save_image(val, img)

            for j in range(y.size(0)):
                _id = idx * y.size(0) + j

                local_data = x[j].squeeze(0).cpu().numpy()
                local_label = y[j].cpu().numpy()

                wandb_image = wandb_mask(local_data, local_label, config.image.labels)                
                dict_val[f'image_{_id:02d}'] = wandb_image
        
    wandb.log(dict_val)

'''def save_test_as_imgs(loader, folder=CONFIG.PATHS.PREDICTIONS_DIR, time=0, device=DEVICE):
    print("=> Saving test images ...")
    
    dict_val = {}
    
    with torch.no_grad(): 
        for idx, (_, y) in enumerate(loader):
            y = y.to(device)
            val = (y / y.max()).unsqueeze(1)
            for j in range(y.size(0)):
                img = f"{folder}{time}_test_i{idx:02d}_p{j:02d}.png"
                save_image(val, img)
                dict_val[f'test_i{idx:02d}_p{j:02d}'] = wandb.Image(img)
    
    wandb.log(dict_val)

def save_ellipse_pred_as_imgs(loader, model, epoch, dict_eval, folder=CONFIG.PATHS.SUBMISSIONS_DIR, time=0, device=DEVICE):
    print("=> Saving ellipses as images ...")
    
    model.eval()
    
    with torch.no_grad():    
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)
            preds_labels = torch.argmax(model(x), 1)
            preds_labels = torch.tensor(fit_ellipses_on_image(preds_labels)).float().to(DEVICE)
            preds_labels = label_to_pixel(preds_labels)
            
            img = folder + f"{time}_elli_e{epoch}_i{idx}.png"
            save_image(preds_labels, img)
            dict_eval[f'elli_prediction_i{idx}'] = wandb.Image(img)
            
    model.train()
    wandb.log(dict_eval)

def save_ellipse_validation_as_imgs(loader, folder=CONFIG.PATHS.PREDICTIONS_DIR, time=0, device=DEVICE):
    print("=> Saving validation ellipses ...")
    
    dict_val = {}
    
    with torch.no_grad(): 
        for idx, (_, y) in enumerate(loader):
            y = torch.tensor(fit_ellipses_on_image(y)).to(device)
            val = label_to_pixel(y)
            img = f"{folder}{time}_elli_val_i{idx:02d}.png"
            save_image(val, img)
            dict_val[f'elli_validation_i{idx:02d}'] = wandb.Image(img)
    
    wandb.log(dict_val)

def save_confidence_as_imgs(loader, model, epoch, dict_eval, folder=CONFIG.PATHS.SUBMISSIONS_DIR, time=0, device=DEVICE):
    print("=> Saving confidence of prediction as images ...")
    
    model.eval()
    
    with torch.no_grad():    
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)
            probs = model(x)
            lesion_confidence = get_confidence_of_prediction(probs)
            
            img = folder + f"{time}_confidence_e{epoch}_i{idx}.png"
            save_image(lesion_confidence, img)
            dict_eval[f'confidence_i{idx}'] = wandb.Image(img)
            
    model.train()'''