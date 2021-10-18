import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_validation_as_imgs,
    get_weights,
)
from loss import (
    ce_loss,
    dice_loss,
    jaccard_loss,
    tversky_loss,
    bce_loss,

)

# Hyperparameters etc.
LEARNING_RATE = 3e-4
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
PARENT_DIR = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/"
TRAIN_IMG_DIR = PARENT_DIR + "train/phantom/"
TRAIN_MASK_DIR = PARENT_DIR + "train/mask/"
VAL_IMG_DIR = PARENT_DIR + "val/phantom/"
VAL_MASK_DIR = PARENT_DIR + "val/mask/"
SUB_IMG_DIR = PARENT_DIR + "test/"
SUB_MASK_DIR = PARENT_DIR + "submission/"
PREDICTIONS_DIR = PARENT_DIR + "predictions/"

torch.manual_seed(19)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #loss = loss_fn(predictions, targets)
            loss = loss_fn(targets, predictions)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():

    train_transforms = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=0.,std=1.,max_pixel_value=1.),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=0.,std=1.,max_pixel_value=1.),
            ToTensorV2(),
        ],
    )

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
    #weights = get_weights(TRAIN_MASK_DIR, MASK_LABELS, device=DEVICE, multiplier = [100, 300, 200, 200])
    #loss_fn = bce_loss()#nn.CrossEntropyLoss(weight=weights)
                         # Initial arguments were:      1e-5,           0.9,          True,              0.0001
                         # Second iteration:            1e-2,           0.9,          True,              0.0001
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=0.0001)
    BEGIN = datetime.now().strftime("%Y%m%d_%H%M%S")

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    #else :
        #print("MODEL SUMMARY")
        #summary(model, (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), BATCH_SIZE, DEVICE)

    check_accuracy(val_loader, model, MASK_LABELS, time=BEGIN, device=DEVICE)

    save_validation_as_imgs(val_loader, folder = PREDICTIONS_DIR, device = DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print('================================================================')
        print('BEGINNING EPOCH', epoch, ':')
        print('================================================================')        
        
        train_fn(train_loader, model, optimizer, dice_loss, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, MASK_LABELS, time=BEGIN, device=DEVICE)

        if epoch % 5 == 0 :
            # print some examples to a folder
            save_predictions_as_imgs(val_loader, model, epoch, folder = PREDICTIONS_DIR, device=DEVICE)

if __name__ == "__main__":
    main()