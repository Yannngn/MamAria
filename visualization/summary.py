import os
from torchinfo import summary
import torch
from model import UNET
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 15
IMAGE_HEIGHT = 512  # 256 originally
IMAGE_WIDTH = 301  # 98 originally
IMAGE_CHANNELS = 1
MASK_CHANNELS = 1
MASK_LABELS = 4
MAX_LAYER_SIZE = 1024
MIN_LAYER_SIZE = 64
PATH = '/'.join(os.path.abspath(__file__).split('\\')[:-2])+'/'

def main():

    config_defaults = {'batch_size': BATCH_SIZE,'max_layer_size': MAX_LAYER_SIZE,'min_layer_size': MIN_LAYER_SIZE}

    wandb.init(
        project = 'summary',
        #entity=PROJECT_TEAM,
        #group='experiment-1',
        config=config_defaults)

    config = wandb.config

    model = UNET(in_channels = IMAGE_CHANNELS, classes = MASK_LABELS, config = config).to(DEVICE)
    with open("summary.txt", "w", encoding="utf-8") as f:
        print(summary(model=model, input_size=(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE), file=f)

if __name__ == '__main__':
    main()
