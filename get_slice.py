import os
from shutil import copyfile
from PIL import Image
import numpy as np

PATH = "C:/Users/Yann/Documents/GitHub/LesionInserter_Data/spiculated_02/"
OUT = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/"
IMG_PATH = OUT + "phantom/"
MSK_PATH = OUT + "mask/"
NAME = "spiculated_02"

def main(path = PATH, name = NAME, get_slice = False, crop = False, output = OUT):
    if get_slice:
        get_slices(PATH, name, OUT)
        
    if crop:
        cropper(IMG_PATH, MSK_PATH)
                                
def get_slices(path, name, out):
    for root, dirs, filenames in os.walk(PATH):
        for folder in dirs:
            if folder[-4:] == "crop":
                copyfile(PATH + folder + "/_recon_slice_317.tiff", IMG_PATH + name + "_" + folder + ".tiff")
            if folder[-4:] == "mask":
                copyfile(PATH + folder + "/_317.png", MSK_PATH + name + "_" + folder + ".png")
                    
def cropper(img, masks):
    phant = Image.open(img + os.listdir(img)[0])
    img_height = phant.size[1]

    for root, dirs, files in os.walk(masks):
        for file in files:
            if file[-4:] == ".png":
                mask = Image.open(masks + file)
                w, h = mask.size
                mask.crop((0, h-img_height, w, h)).save(masks + file)
                
if __name__ == "__main__":
    main(PATH, NAME, True, True)