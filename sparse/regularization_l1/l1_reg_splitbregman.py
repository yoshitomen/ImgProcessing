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
    I_origin = cv2.cvtColor(img_load, cv2.COLOR_RGB2GRAY)

    I_noise = lib.add_noise(I_origin)

    LAMBDA = 50 ## 正則化パラメータ
    mean = np.array(I_origin).flatten().mean()   
    THRE  = mean #255#0 #255:background=white, 0:background=black.
    print(f"Mean:{THRE}")

    I_reconst = SplitBregman_L1(I_noise,LAMBDA,THRE)

    plt.figure(figsize=[9,3])
    plt.subplot(1, 3, 1)
    plt.gray()
    plt.imshow(I_origin)
    plt.title("Original")
    plt.subplot(1,3,2)
    plt.gray()
    plt.imshow(I_noise)
    plt.title("Noisy")
    plt.subplot(1,3,3)
    plt.gray()
    plt.imshow(I_reconst)
    plt.title("Reconstructed")
    plt.savefig(this_file_path+"/data/L1_reconst.png")
    plt.close()

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(I_noise[50,:])
    plt.ylim(0,250)
    plt.subplot(2,1,2)
    plt.plot(I_reconst[50,:])
    plt.ylim(0,250)
    plt.savefig(this_file_path+"/data/comp_line.png")
    plt.close()

    LAMBDA_LIST = [5, 30, 50, 100]
    #THRE  = 200
    I_LIST = []
    plt.figure(figsize=[12,9])
    for Lambda in LAMBDA_LIST:
        I_reconst = SplitBregman_L1(I_noise,Lambda,THRE)
        I_LIST.append(I_reconst)

    for i in range(len(LAMBDA_LIST)):
        plt.subplot(2, int(len(LAMBDA_LIST)/2), i+1)
        plt.gray()
        plt.imshow(I_LIST[i])
        plt.title("$\lambda = $" + str(LAMBDA_LIST[i]))
    plt.savefig(this_file_path+"/data/L1_param.png")

def SplitBregman_L1(I_noise,LAMBDA,THRE):
    B = I_noise - THRE
    t = np.sign(B)*B - LAMBDA 
    I_reconst = t*(t>0)*np.sign(B) + THRE
    return I_reconst

if __name__ == "__main__":
    main()