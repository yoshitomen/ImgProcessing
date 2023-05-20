#import cv2
import time
import os
import cv2
import numpy as np
from cv2 import dnn_superres

import Resize_DelMeta as res_photo

def UpConvert():
    start_time = time.time()
    # Create an SR object - only function that differs from c++ code
    sr = dnn_superres.DnnSuperResImpl_create()

    # Read image
    image = cv2.imread('./origin/sample.JPG')

    # Read the desired model
    path = "EDSR_x4.pb"
    sr.readModel(path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 4)

    # Upscale the image
    result = sr.upsample(image)

    # Save the image
    cv2.imwrite("./upconvert/upconvert.JPG", result)
    print(time.time()-start_time)

"""
    dir_name = "origin"
    os.mkdir("upconvert")
    save_dir = "upconvert"
    #original画像のアップコンバート

    start_time = time.time()
    # 画像を読み込む
    files = os.listdir(dir_name)
    for file in files:
        if file == '.DS_Store':continue #Mac特有の隠しファイル対策.
        photo = Image.open(os.path.join(dir_name, file))
        #img = cv2.imread("image.jpg")

        # SR3アルゴリズムで高画質化を適用する
        photo_upconvert = SR3(photo, scale=4)

        # 画像を保存する
        photo_upconvert.save(os.path.join(save_dir, file))
        #cv2.imwrite(f'result_image_{file}.jpg", photo_upconvert')
    print(time.time()-start_time)
"""

if __name__ == "__main__":
    UpConvert()