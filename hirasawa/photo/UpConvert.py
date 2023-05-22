import Resize_DelMeta
import time
import os
import cv2
import numpy as np
from cv2 import dnn_superres

import Resize_DelMeta as res_photo

def UpConvert():
    start_time = time.time()

    sr = dnn_superres.DnnSuperResImpl_create()
    image = cv2.imread('./origin/sample.jpg')
    path = "EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 4)
    result = sr.upsample(image)

    cv2.imwrite("./upconvert/upconvert.jpg", result)
    print(time.time()-start_time)

    Resize_DelMeta.resize(100,100)
    start_time = time.time()
    image = cv2.imread('./resized/sample.jpg')
    result2 = sr.upsample(image)
    result3 = sr.upsample(result2)
    cv2.imwrite("./upconvert/upconvert_resized.jpg", result3)
    print(time.time()-start_time)


if __name__ == "__main__":
    UpConvert()