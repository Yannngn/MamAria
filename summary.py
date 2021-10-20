from torchsummary import summary
import torch
from model import UNET

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 54
IMAGE_HEIGHT = 256  # 256 originally
IMAGE_WIDTH = 98  # 98 originally
IMAGE_CHANNELS = 1
MASK_CHANNELS = 1
MASK_LABELS = 4
def main():
    model = UNET(in_channels = IMAGE_CHANNELS, classes = MASK_LABELS).to(DEVICE)
    summary(model, (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), BATCH_SIZE, DEVICE)

if __name__ == '__main__':
    main()
