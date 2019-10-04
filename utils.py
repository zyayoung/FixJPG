import cv2
import numpy as np
import os
from tqdm import tqdm
from keras.metrics import mse

def get_train_data(shape, pad):
    imgs = os.listdir("y")
    x = []
    y = []
    for img in tqdm(imgs):
        xim = cv2.imread('x/'+img)
        yim = cv2.imread('y/'+img)
        xim = np.expand_dims(cv2.cvtColor(xim, cv2.COLOR_BGR2GRAY),-1)
        yim = np.expand_dims(cv2.cvtColor(yim, cv2.COLOR_BGR2GRAY),-1)
        for i in range(0, xim.shape[0] - shape[0], shape[0]-pad*2):
            for j in range(0, xim.shape[1] - shape[1], shape[1]-pad*2):
                x.append(xim[i:i+shape[0],j:j+shape[1]])
                y.append(yim[i+pad:-pad+i+shape[0],j+pad:-pad+j+shape[1]])
            if (xim.shape[1] - shape[1]) % shape[1]-pad*2 and xim.shape[1]>shape[1]:
                x.append(xim[i:i+shape[0],-shape[1]:])
                if pad:
                    y.append(yim[i+pad:-pad+i+shape[0],-shape[1]+pad:-pad])
                else:
                    y.append(yim[i:i+shape[0],-shape[1]:])
        if (xim.shape[0] - shape[0]) % shape[0]-pad*2 and xim.shape[0]>shape[0]:
            for j in range(0, xim.shape[1] - shape[1], shape[1]-pad*2):
                x.append(xim[-shape[0]:,j:j+shape[1]])
                if pad:
                    y.append(yim[-shape[0]+pad:-pad,j+pad:-pad+j+shape[1]])
                else:
                    y.append(yim[-shape[0]:,j:j+shape[1]])
            if (xim.shape[1] - shape[1]) % shape[1]-pad*2 and xim.shape[1]>shape[1]:
                x.append(xim[-shape[0]:,-shape[1]:])
                if pad:
                    y.append(yim[-shape[0]+pad:-pad,-shape[1]+pad:-pad])
                else:
                    y.append(yim[-shape[0]:,-shape[1]:])

    return np.array(x)/255., np.array(y)/255.

def get_full_data(shape):
    imgs = os.listdir("y")
    x = []
    y = []
    for img in tqdm(imgs):
        xim = cv2.imread('x/'+img)
        yim = cv2.imread('y/'+img)
        x.append(cv2.resize(xim[-shape[0]:,-shape[1]:], shape, interpolation=cv2.INTER_CUBIC))
        y.append(cv2.resize(yim[-shape[0]:,-shape[1]:], shape, interpolation=cv2.INTER_CUBIC))
    return np.array(x), np.array(y)

def get_raw_data():
    imgs = os.listdir("y")
    x = []
    y = []
    for img in tqdm(imgs):
        xim = cv2.imread('x/'+img)
        yim = cv2.imread('y/'+img)
        xim = np.expand_dims(cv2.cvtColor(xim, cv2.COLOR_BGR2GRAY),-1)
        yim = np.expand_dims(cv2.cvtColor(yim, cv2.COLOR_BGR2GRAY),-1)
        x.append(xim)
        y.append(yim)
    return x, y

def get_name_data():
    imgs = os.listdir("y")
    return imgs

def extend_sym(x):
    x = np.concatenate([x.transpose(0,2,1,3), x])
    x = np.concatenate([x[:,:,::-1], x])
    x = np.concatenate([x[:,::-1,:], x])
    return x

from keras import backend as K

def PSNR(y_true, y_pred):
    return 10.0 * K.log(1 / (mse(y_true, y_pred) + 1e-8)) / 2.3025851
