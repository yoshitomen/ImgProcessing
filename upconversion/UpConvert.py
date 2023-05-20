import Resize_DelMeta
import time
import os
import cv2
import glob
import numpy as np
from cv2 import dnn_superres

import Resize_DelMeta as res_photo
file_path = os.path.dirname(os.path.abspath(__file__))
model_path = f"{file_path}/Model/EDSR_x4.pb" #PATH from this file

def UpConvert_test():
    start_time = time.time()

    sr = dnn_superres.DnnSuperResImpl_create()
    image = cv2.imread(f'{file_path}/data/origin/sample.jpg')
    sr.readModel(model_path)
    sr.setModel("edsr", 4)
    result = sr.upsample(image)

    cv2.imwrite(f"{file_path}/data/upconvert_test/upconvert.jpg", result)
    print(time.time()-start_time)

    Resize_DelMeta.resize(100,100)
    start_time = time.time()
    image = cv2.imread(f'{file_path}/data/resized/sample.jpg')
    result2 = sr.upsample(image)
    result3 = sr.upsample(result2)
    cv2.imwrite(f"{file_path}/data/upconvert_test/upconvert_resized.jpg", result3)
    print(time.time()-start_time)

def UpConvert():
    start_time = time.time()

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4)
    Resize_DelMeta.resize_mag(1)
    files = glob.glob(f"{file_path}/data/resized/*")
    for file in files:
        if file == '.DS_Store':continue #Mac特有の隠しファイル対策.
        if not (('.jpg' in file) or ('.png' in file)):continue 
        file_name = os.path.basename(file)
        image = cv2.imread(file)
        result = sr.upsample(image)
        print(file_name)
        cv2.imwrite(f"{file_path}/data/upconvert/{file_name}", result)
    #Resize_DelMeta.resize(225,225,"data/upconvert/origin","data/upconvert/origin")
    print(time.time()-start_time)


if __name__ == "__main__":
    #UpConvert_test()
    UpConvert()