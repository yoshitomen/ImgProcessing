#pip install Pillow
import os
from PIL import Image
file_path = os.path.dirname(os.path.abspath(__file__))

"""======================================
@brief resize specification
======================================"""
def resize(x=100,y=100,dir_name=f"{file_path}/data/origin", resize_dir=f"{file_path}/data/resized"):
    #dir_name = "origin"
    #os.mkdir("resized")
    resize_size = (x,y)
    #resize_dir = "resized"
    files = os.listdir(dir_name)
    for index, file in enumerate(files):
        if file == '.DS_Store':continue #Mac特有の隠しファイル対策.
        if not (('.jpg' in file) or ('.png' in file)):continue
        photo = Image.open(os.path.join(dir_name, file))
        print(f'original_size = [width={photo.width}], height={photo.height}]')
        photo_resize = photo.resize(resize_size)
        print(f'resized_size = [width={photo_resize.width}, height={photo_resize.height}]')
        photo_resize.save(os.path.join(resize_dir, file))

"""======================================
@brief resize magnification
======================================"""
def resize_mag(mag=4, dir_name=f"{file_path}/data/origin", resize_dir=f"{file_path}/data/resized"):
    files = os.listdir(dir_name)
    for index, file in enumerate(files):
        if file == '.DS_Store':continue #Mac特有の隠しファイル対策.
        if not (('.jpg' in file) or ('.png' in file)):continue 
        photo = Image.open(os.path.join(dir_name, file))
        print(f'original_size = [width={photo.width}], height={photo.height}]')
        photo_resize = photo.resize((photo.width*mag, photo.height*mag))
        print(f'resized_size = [width={photo_resize.width}, height={photo_resize.height}]')
        photo_resize.save(os.path.join(resize_dir, file))

def resize_mag_t(image, mag=4):
    return image.resize((image.width*mag, image.height*mag))
    
if __name__ == "__main__":
    #resize(6000,4000)
    resize_mag(4)
