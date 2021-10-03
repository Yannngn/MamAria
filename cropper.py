import os
from PIL import Image

DIR = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/mask/"
PHA = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/phantom/"

phant = Image.open(PHA + os.listdir(PHA)[0])
print(os.listdir(PHA)[0])
img_height = phant.size[1]

for root, dirs, files in os.walk(DIR):
    for file in files:
        if file[-4:] == ".png":
            mask = Image.open(DIR + file)
            w, h = mask.size
            mask.crop((0, img_height-256, w, h)).save(DIR + file)