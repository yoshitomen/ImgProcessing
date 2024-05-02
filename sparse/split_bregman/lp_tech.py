import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image

import sys,os


def add_noise(I_t):
    mu = np.mean(I_t)
    sigma = np.std(I_t)
    dB = 3
    I_noise = 10**(-dB/20)*np.reshape([random.gauss(mu, sigma) for i in range(np.size(I_t))], np.shape(I_t))
    I = I_t + I_noise
    max_I  = np.max(I)
    min_I = np.min(I)
    I = np.round((I - min_I)*255/(max_I - min_I))
    return I

this_file_path = os.path.dirname(sys.argv[0])
img_path = this_file_path+"/origin_lp.jpg"
img_load = cv2.imread(img_path)
I_t = cv2.cvtColor(img_load, cv2.COLOR_RGB2GRAY)
f = add_noise(I_t)
[X_N,Y_N] = np.shape(f)
print(X_N,Y_N)
plt.subplot(1,2,1)
plt.imshow(I_t, cmap="gray")
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(f, cmap="gray")
plt.title("Noisy")
plt.show()

def add_noise(I_t):
    mu = np.mean(I_t)
    sigma = np.std(I_t)
    dB = 3
    I_noise = 10**(-dB/20)*np.reshape([random.gauss(mu, sigma) for i in range(np.size(I_t))], np.shape(I_t))
    I = I_t + I_noise
    max_I  = np.max(I)
    min_I = np.min(I)
    I = np.round((I - min_I)*255/(max_I - min_I))
    return I


def Gauss_Saidel(u, d_x, d_y, b_x, b_y, f, MU, LAMBDA):
    U = np.hstack([u[:,1:X_N], np.reshape(u[-1,:],[Y_N,1] )]) + np.hstack([np.reshape(u[0,:],[Y_N,1]), u[:,0:Y_N-1]]) \
       + np.vstack([u[1:X_N,:], np.reshape(u[:,-1],[1,X_N] )]) + np.vstack([np.reshape(u[:,0],[1,X_N] ), u[0:X_N-1,:]])
    D = np.vstack([np.reshape(d_x[:,0],[1,X_N] ), d_x[0:Y_N-1,:]]) - d_x \
       + np.hstack([np.reshape(d_y[0,:],[Y_N,1] ), d_y[:,0:X_N-1]]) - d_y
    B = -np.vstack([np.reshape(b_x[:,0],[1,X_N] ), b_x[0:Y_N-1,:]]) + b_x \
       - np.hstack([np.reshape(b_y[0,:],[Y_N,1] ), b_y[:,0:X_N-1]]) + b_y
    G = LAMBDA/(MU + 4*LAMBDA)*(U+D+B) + MU/(MU + 4*LAMBDA)*f
    return G
    

def shrink(x,y):
    t = np.abs(x) - y
    S = np.sign(x)*(t > 0) * t
    return S

def main():
    ## Load Image
    #img_path = "Blood2.jpg"
                 
    CYCLE = 100
    MU = 5.0*10**(-2)
    LAMBDA = 1.0*10**(-2)
    TOL = 5.0*10**(-1)

    ## Initialization
    u = f
    d_x = np.zeros([X_N,Y_N])
    d_y = np.zeros([X_N,Y_N])
    b_x = np.zeros([X_N,Y_N])
    b_y = np.zeros([X_N,Y_N])

    for cyc in range(CYCLE):
        u_n = Gauss_Saidel(u,d_x,d_y, b_x ,b_y,f, MU,LAMBDA)
        Err = np.max(np.abs(u_n[2:X_N-2,2:Y_N-2] - u[2:X_N-2,2:Y_N-2]))
        if np.mod(cyc,10)==0:
            print([cyc,Err])
        if Err < TOL:
            break
        else:
            u = u_n
            nablax_u = np.vstack([u[1:X_N,:], np.reshape(u[:,-1],[1,X_N] )]) - u 
            nablay_u = np.hstack([u[:,1:X_N], np.reshape(u[-1,:],[Y_N,1] )]) - u 
            d_x = shrink(nablax_u + b_x, 1/LAMBDA)
            d_y = shrink(nablay_u + b_y, 1/LAMBDA)
            b_x = b_x + (nablax_u - d_x)
            b_y = b_y + (nablay_u - d_y)
    
    
    ## plot figure
    plt.figure()
    plt.subplot(1,2,1)
    plt.gray()
    plt.imshow(f, cmap="gray")
    plt.title('Noisy')
    #plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.gray()
    plt.imshow(np.round(u), cmap="gray")
    #x1, y1 = [0,X_N], [50,50]
    #plt.plot(x1, y1)
    plt.title('Reconstructed')
    #plt.axis("off")
    plt.show()
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(f[50,:])
    plt.subplot(2,1,2)
    plt.plot(u[50,:])
    plt.show()

if __name__ == "__main__":
    main()