import os
from torchinfo import summary
import torch
from model import UNET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 54
IMAGE_HEIGHT = 256  # 256 originally
IMAGE_WIDTH = 98  # 98 originally
IMAGE_CHANNELS = 1
MASK_CHANNELS = 1
MASK_LABELS = 4
PATH = '/'.join(os.path.abspath(__file__).split('\\')[:-2])+'/'
def main():
    model = UNET(in_channels = IMAGE_CHANNELS, classes = MASK_LABELS).to(DEVICE)
    with open("summary.txt", "w", encoding="utf-8") as f:
        print(summary(model=model, input_size=(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE), file=f)

if __name__ == '__main__':
    main()
