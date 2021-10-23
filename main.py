import wandb

import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from datetime import datetime

from model import UNET
from early_stopping import EarlyStopping
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_validation_as_imgs,
    get_weights,
    print_and_save_results
)

torch.manual_seed(19)

# Hyperparameters etc.
PROJECT_NAME = "lesion_segmentation_4285_50_02"
LEARNING_RATE = 3e-4 ### Begin with 3e-4, 96.41% now 8e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 50
NUM_EPOCHS = 1000
NUM_WORKERS = 12
IMAGE_HEIGHT = 256  # 256 originally
IMAGE_WIDTH = 98  # 98 originally
IMAGE_CHANNELS = 1
MASK_CHANNELS = 1
MASK_LABELS = 4
PIN_MEMORY = True
LOAD_MODEL = False
PARENT_DIR = "data/"
TRAIN_IMG_DIR = PARENT_DIR + "train/phantom/"
TRAIN_MASK_DIR = PARENT_DIR + "train/mask/"
VAL_IMG_DIR = PARENT_DIR + "val/phantom/"
VAL_MASK_DIR = PARENT_DIR + "val/mask/"
SUB_IMG_DIR = PARENT_DIR + "test/"
SUB_MASK_DIR = PARENT_DIR + "submission/"
PREDICTIONS_DIR = PARENT_DIR + "predictions/"

#sweep_id = wandb.sweep('sweep_config.yaml', project=PROJECT_NAME)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    config_defaults = {
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
    }

    wandb.init(
        project = PROJECT_NAME,
        entity='tail-upenn',
        #group='experiment-1',
        config=config_defaults)

    config = wandb.config
    closs = 0
    for _, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        wandb.log({"batch loss":loss.item()})
        closs += loss.item()
    wandb.log({"loss":closs/config.batch_size}) 
    ### IF doesnt work remove here
    return loss.item()

def validate_fn(loader, model, loss_fn):
    loop = tqdm(loader)
    model.eval()
    
    for _, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    model.train()
    
    return loss.item()

def main():

    train_transforms = A.Compose([ToTensorV2(),],)

    val_transforms = A.Compose([ToTensorV2(),],)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    model = UNET(in_channels = IMAGE_CHANNELS, classes= MASK_LABELS).to(DEVICE)
    weights = get_weights(TRAIN_MASK_DIR, MASK_LABELS, DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight = weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    BEGIN = datetime.now().strftime("%Y%m%d_%H%M%S")

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    # else :
    #     print("MODEL SUMMARY")
    #     summary(model, (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), BATCH_SIZE, DEVICE)

    check_accuracy(val_loader, model, MASK_LABELS, DEVICE)

    save_validation_as_imgs(val_loader, folder = PREDICTIONS_DIR, device = DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    stopping = EarlyStopping(patience = 15, mode = 'min', wait=20)

    for epoch in range(NUM_EPOCHS):
        print('================================================================')
        print('BEGINNING EPOCH', epoch, ':')
        print('================================================================')        
        
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        save_checkpoint(checkpoint)

        # check accuracy
        accuracies = check_accuracy(val_loader, model, MASK_LABELS, DEVICE)

        val_loss = validate_fn(val_loader, model, loss_fn)
        scheduler.step(val_loss)
        metrics = accuracies[2]
        dict_log = {"val_loss":val_loss,
                    "accuracy":metrics[0],
                    "label_0_accuracy":metrics[1][0],
                    "label_1_accuracy":metrics[1][1],
                    "label_2_accuracy":metrics[1][2],
                    "label_3_accuracy":metrics[1][3],
                    "label_0_recall":metrics[2][0],
                    "label_1_recall":metrics[2][1],
                    "label_2_recall":metrics[2][2],
                    "label_3_recall":metrics[2][3],
                    # "label_0_f1":metrics[3][0],
                    # "label_1_f1":metrics[3][1],
                    # "label_2_f1":metrics[3][2],
                    # "label_3_f1":metrics[3][3],
                    # "label_0_iou":metrics[4][0],
                    # "label_1_iou":metrics[4][1],
                    # "label_2_iou":metrics[4][2],
                    # "label_3_iou":metrics[4][3]
                   }
        
        wandb.log(dict_log) 
        print_and_save_results(accuracies[0], accuracies[1], metrics, train_loss, val_loss, BEGIN)

        if epoch % 5 == 0 :
            # print some examples to a folder
            save_predictions_as_imgs(val_loader, model, epoch, folder = PREDICTIONS_DIR, device = DEVICE)

        stopping(val_loss, checkpoint, checkpoint_path=f"checkpoints/{BEGIN}_best_checkpoint.pth.tar", epoch = epoch)
        
        if stopping.early_stop:
            print("Early Stopping ...")
            #wandb.agent(sweep_id, train_fn)
            wandb.finish()
            break

        wandb.finish()

if __name__ == "__main__":
    main()