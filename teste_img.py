import cv2
import os
import numpy as np
from PIL import Image
import torch
import albumentations as A
import imageio

img_path = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/phantom/Richards_888076.1.555465657049.20210203023509248-crop.tiff"
mask_path = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/mask/Richards_888076.1.555465657049.20210203023509248_mask.png"
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
            A.Normalize(mean=[0.0, 0.0, 0.0],std=[1.0, 1.0, 1.0],max_pixel_value=255.0),
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

def __getitem__(image_dir, mask_dir, transform, index):
    images = os.listdir(image_dir)

    img_path = os.path.join(image_dir, images[index])
    mask_path = os.path.join(mask_dir, images[index].replace("-crop.tiff", "_mask.png"))

    image = imageio.imread(img_path)    
    image = (image >> 6).astype('uint8')
    image = Image.fromarray(image).convert("RGB")
    image.save("C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/test/output_pre.png")
   

    #mask = imageio.imread(mask_path) 
    #mask = (mask << 3).astype('uint8')
    #mask = Image.fromarray(mask).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    mask.save("C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/test/mask_output_pre.png")

    img_new, mask_new = image, mask

    if transform is not None:
        image_np = np.array(image)
        mask_np = np.array(mask) 
        augmentations = transform(image = image_np, mask = mask_np)
        img_new = augmentations["image"]
        mask_new = augmentations["mask"]
        img_new = torch.from_numpy(img_new).float()
    else:
        img_new = torch.from_numpy(img_new).float()

    img_new = img_new.permute(2, 0, 1).contiguous()
#       norm = tvtransforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
#       img_new = norm(img_new)

    #mask_new = mask_to_class_rgb(mapping, mask_new)
    #class_to_rgb(mapping, mask_new)
    #Image.fromarray(mask_new.numpy()).save(PARENT_DIR + "test/mask_rgb.tiff")
    #mask_new = mask_new.float()
    mask_new = torch.from_numpy(mask_new).float()

    return img_new, mask_new

def class_to_rgb(mapping, mask):
    mask = torch.from_numpy(mask)
    print(mask.shape)
    mask = torch.squeeze(mask)
    print(mask.shape)
    print(torch.unique(mask))
    return mask




def mask_to_class_rgb(mapping, mask):
    #print('----mask->rgb----')
    mask = torch.from_numpy(mask)
    mask = torch.squeeze(mask)  # remove 1

    # check the present values in the mask, 0 and 255 in my case
    #print('unique values rgb    ', torch.unique(mask)) 
    # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

    class_mask = mask
    class_mask = class_mask.permute(2, 0, 1).contiguous()
    h, w = class_mask.shape[1], class_mask.shape[2]
    mask_out = torch.empty(h, w, dtype=torch.float)

    for k in mapping:
        idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
                
        validx = (idx.sum(0) == 3)          
        mask_out[validx] = torch.tensor(mapping[k], dtype=torch.float)

    # check the present values after mapping, in my case 0, 1, 2, 3
    #print('unique values mapped ', torch.unique(mask_out))
    # -> unique values mapped  tensor([0, 1, 2, 3])

    return mask_out

if __name__ == "__main__":
    retornos = __getitem__(IMG_DIR, MASK_DIR, None, 0)
    print(np.unique(retornos[0]))
