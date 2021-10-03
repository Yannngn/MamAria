import os
from PIL import Image

DIR = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/mask/"

for root, dirs, files in os.walk(DIR):
    for filename in files:
        if filename[-4:] == ".png":
            mask = Image.open(os.path.join(DIR, filename))
            w, h = mask.size
            mask.crop((0, h-256, w, h)).save(os.path.join(DIR, filename))