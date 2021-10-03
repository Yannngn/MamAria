import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.modules import loss
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from soft_IoU_loss import SoftIoULoss
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 25
NUM_EPOCHS = 1000
NUM_WORKERS = 12
IMAGE_HEIGHT = 256  # 256 originally
IMAGE_WIDTH = 98  # 98 originally
PIN_MEMORY = True
LOAD_MODEL = False
PARENT_DIR = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/"
TRAIN_IMG_DIR = PARENT_DIR + "phantom/"
TRAIN_MASK_DIR = PARENT_DIR + "mask/"
VAL_IMG_DIR = PARENT_DIR + "phantom/"
VAL_MASK_DIR = PARENT_DIR + "mask/"
SUB_IMG_DIR = PARENT_DIR + "test/"
SUB_MASK_DIR = PARENT_DIR + "submission/"
PREDICTIONS_DIR = PARENT_DIR + "predictions/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        #print("target shape", targets.shape)
        data = data.to(device=DEVICE)
        #print(targets.numpy().shape)
        #targets = targets.float().unsqueeze(1).to(device=DEVICE)
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

def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.0, 0.0, 0.0],std=[1.0, 1.0, 1.0],max_pixel_value=255.0),
            #ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0],std=[1.0, 1.0, 1.0],max_pixel_value=255.0),
            #ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, classes=4).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = SoftIoULoss()
    #loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder = PREDICTIONS_DIR, device=DEVICE)

        print(epoch)

if __name__ == "__main__":
    main()