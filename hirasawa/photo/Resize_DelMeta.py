#pip install Pillow
import os
from PIL import Image

def resize():
    dir_name = "origin"
    #os.mkdir("resized")
    resize_size = (6000,4000)
    new_dir_name = "resized"
    files = os.listdir(dir_name)
    for file in files:
            if file == '.DS_Store':continue #Mac特有の隠しファイル対策.
            photo = Image.open(os.path.join(dir_name, file))
            photo_resize = photo.resize(resize_size)
            photo_resize.save(os.path.join(new_dir_name, file))

if __name__ == "__main__":
    resize()
