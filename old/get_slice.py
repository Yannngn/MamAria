import os
from shutil import copyfile
from PIL import Image


OUT = "C:/Users/Yann/Documents/GitHub/PyTorch_Seg/data/"
IMG_PATH = OUT + "phantom/"
MSK_PATH = OUT + "mask/"
FILES = ["null_9_0000_00_00",
         "sphere_0_4285_50_02",
         "spiculated_1_4285_50_02",
         "spiculated_2_4285_50_02",]
PATH = "C:/Users/Yann/Documents/GitHub/LesionInserter-data/"

def main(path = PATH, files = FILES, img_path = IMG_PATH, mask_path = MSK_PATH, get_slice = False, crop = False):
    for file in files:
        if get_slice:
            get_slices(path+file+'/', file, img_path, mask_path)
            
        if crop:
            cropper(img_path, mask_path)
                                
def get_slices(path, name, img_path = IMG_PATH, mask_path = MSK_PATH):
    a, j, r, s = 0, 0, 0, 0
    for _, dirs, _ in os.walk(path):
        for folder in dirs:
            if folder[-4:] == "crop":
                family = folder.split('_')[0]

                if family == 'Alvarado':
                    image = f'_{a:03}'
                    a += 1
                elif family == 'Jhones':
                    image = f'_{j:03}'
                    j += 1
                elif family == 'Richards':
                    image = f'_{r:03}'
                    r += 1
                elif family == 'Stark':
                    image = f'_{s:03}'
                    s += 1
                else:
                    raise ValueError('error, unknown family', family)
                
                copyfile(path + folder + "/_recon_slice_317.tiff", img_path + family + "_" + name + image + "-crop.tiff")

                copyfile(path + folder[:-5] + "_mask/_317.png", mask_path + family + "_" + name + image + "_mask.png")

                    
def cropper(img, masks):
    phant = Image.open(img + os.listdir(img)[0])
    img_height = phant.size[1]

    for _, _, files in os.walk(masks):
        for file in files:
            if file[-4:] == ".png":
                mask = Image.open(masks + file)
                w, h = mask.size
                mask.crop((0, h - img_height, w, h)).save(masks + file)
                
if __name__ == "__main__":
    main(get_slice = True, crop = True)