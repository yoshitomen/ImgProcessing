"""
Reference
L1正則化(Split Bregmanによる最適化).
L1ノルムの最小化により基準値を多く含む解になる
https://lp-tech.net/articles/CY2Kn
"""

import sys,os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(sys.argv[0])))
import lib.lib as lib

def main():

    this_file_path = os.path.dirname(sys.argv[0])
    img_path = this_file_path+"/data/origin_l1norm.jpg"
    img_load = cv2.imread(img_path)
    I_t = cv2.cvtColor(img_load, cv2.COLOR_RGB2GRAY)

    I = lib.add_noise(I_t)

    LAMBDA = 50 ## 正則化パラメータ
    THRE  = 200 ## だいたい画素値200 くらいが background

    B = I - THRE
    t = np.sign(B)*B - LAMBDA 
    I_reconst = t*(t>0)*np.sign(B) + THRE

    plt.figure(figsize=[9,3])
    plt.subplot(1, 3, 1)
    plt.gray()
    plt.imshow(I_t)
    plt.title("Original")
    plt.subplot(1,3,2)
    plt.gray()
    plt.imshow(I)
    plt.title("Noisy")
    plt.subplot(1,3,3)
    plt.gray()
    plt.imshow(I_reconst)
    plt.title("Reconstructed")
    plt.savefig(this_file_path+"/data/L1_reconst.png")
    plt.close()

    LAMBDA_LIST = [5, 30, 50, 100]
    THRE  = 200
    I_LIST = []
    plt.figure(figsize=[12,9])
    for Lambda in LAMBDA_LIST:
        B = I - THRE
        t = np.sign(B)*B - Lambda
        I_reconst = t*(t>0)*np.sign(B) + LAMBDA
        I_LIST.append(I_reconst)

    for i in range(len(LAMBDA_LIST)):
        plt.subplot(2, int(len(LAMBDA_LIST)/2), i+1)
        plt.gray()
        plt.imshow(I_LIST[i])
        plt.title("$\lambda = $" + str(LAMBDA_LIST[i]))
    plt.savefig(this_file_path+"/data/L1_param.png")

if __name__ == "__main__":
    main()