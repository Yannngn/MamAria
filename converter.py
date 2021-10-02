import os
from PIL import Image




def convert_imgs(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            jpg_exists = os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg")

            if jpg_exists is False:
                if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
                    outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                    convert_tiff(root, name, outfile)
                elif os.path.splitext(os.path.join(root, name))[1].lower() == ".png":
                    outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                    convert_png(root, name, outfile)

def convert_tiff(root, name, outfile):
    try:
        im = Image.open(os.path.join(root, name))
        print("Generating jpeg for %s" % name)
        im.thumbnail(im.size)
        im.save(outfile, "JPEG", quality=100)
    except Exception as e:
        print(e)

def convert_png(root, name, outfile):
    try:
        im = Image.open(os.path.join(root, name))
        print("Generating jpeg for %s" % name)
        im.thumbnail(im.size)
        im.save(outfile, "JPEG", quality=100)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    path = "C:\\Users\Yann\\Documents\\GitHub\\PyTorch_Seg\\data"
    convert_imgs(path)