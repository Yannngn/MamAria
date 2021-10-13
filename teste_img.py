import cv2
import os
import numpy as np
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image

img_path = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/phantom/sphere_Richards_888076.1.555465657049.20210203023509248-crop.tiff"
mask_path = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/mask/sphere_Richards_888076.1.555465657049.20210203023509248_mask.png"
mapping = {(0, 0, 0): 0, # 0 = no risk / background
           (1, 1, 1): 1, # 1 = low risk
           (2 ,2, 2): 2, # 2 = medium risk
           (3, 3, 3): 3} # 3 = high risk  

IMAGE_HEIGHT = 256  # 256 originally
IMAGE_WIDTH = 98  # 98 originally

PARENT_DIR = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/"
IMG_DIR = PARENT_DIR + "phantom/"
MASK_DIR = PARENT_DIR + "mask/"

transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.],std=[1.],max_pixel_value=1.),
            ToTensorV2()
        ],
    )

# msk = cv2.imread(mask_path, 0)
# #print(np.unique(msk))

# img = cv2.imread(img_path, 0)
# #print(np.unique(img))

# image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
# mask = np.array(Image.open(mask_path))

# mask_tensor = torch.from_numpy(np.array(mask, dtype=np.uint8))
# mask_class = mask_tensor.view(mask.shape[0], mask.shape[1], 1).expand(-1, -1, 3)
# print(np.unique(mask_class.shape, mask.shape))

def __getitem__(image_dir, mask_dir, index):
    #images = os.listdir(image_dir)
    # Get Image and corresponding Mask paths
    # img_path =  os.path.join(image_dir, images[index])
    # mask_path =  os.path.join(mask_dir, images[index].replace("-crop.tiff", "_mask.png"))

    # Convert Image and Mask to numpy
    image = np.array(Image.open(image_dir), dtype=np.float32)
    mask = np.array(Image.open(mask_dir), dtype=np.float32)

    # Seting Image and Mask pixels to 0 1 interval
    image = (image / image.max())
    mask = mask / 3

    # Applying transforms (Normalize and ToTensorV2)
    if transform is not None:
        augmentations = transform(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]

    # If not applyed Image and Mask outputs will be Numpy and not Tensor

    save_image(image, 'image_test.tiff')
    save_image(mask, 'mask_test.png')

    return image, mask

if __name__ == "__main__":
    retornos = __getitem__(img_path, mask_path, 0)
    print(np.unique(retornos[0]))
