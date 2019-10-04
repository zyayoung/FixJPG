import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Input, Lambda, add
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, RMSprop
from keras.utils import multi_gpu_model

def EDSR(input_shape = (48, 48, 3), n_feats = 256, n_resblocks = 32):
    ''' 
        According to the paper scale can be 2,3 or 4. 
        However this code supports scale to be 3 or any of 2^n for n>0
    '''
    def res_block(input_tensor, nf, res_scale = 1.0):
        x = Conv2D(nf, (3, 3), padding='same', activation = 'relu', 
                   activity_regularizer=regularizers.l1(10e-10))(input_tensor)
        x = Conv2D(nf, (3, 3), padding='same', activity_regularizer=regularizers.l1(10e-10))(x)
        x = Lambda(lambda x: x * res_scale)(x)
        x = add([x, input_tensor])
        return x
    inp = Input(shape = input_shape)
    
    x = Conv2D(n_feats, 3, padding='same', activity_regularizer=regularizers.l1(10e-10))(inp)
    conv1 = x
    if n_feats == 256:
        res_scale = 0.1
    else:
        res_scale = 1.0
    for i in range(n_resblocks): x = res_block(x, n_feats, res_scale)
    x = Conv2D(n_feats, 3, padding='same', activity_regularizer=regularizers.l1(10e-10))(x)
    x = add([x, conv1])
    
    sr = Conv2D(input_shape[-1], 1, padding='same', 
                activity_regularizer=regularizers.l1(10e-10))(x)
            
    model = Model(inputs=inp, outputs=sr, name = 'SR')
    return model

#%%
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
#%%
from utils import *
x_train, y_train = get_train_data((48,48), 0)

#%%
print(x_train.shape)
print(y_train.shape)

#%%

model = EDSR(input_shape = (48, 48, 3), n_feats = 64, n_resblocks = 8)
model_multi = model  # multi_gpu_model(model, gpus=1)
model_multi.compile(loss="mse", optimizer=Adam(0.001))
model_multi.fit(x_train, y_train, epochs=6, batch_size=4, validation_split=0.25)
model_multi.compile(loss="mse", optimizer=SGD(0.0001))
model_multi.fit(x_train, y_train, epochs=6, batch_size=4, validation_split=0.25)

model.save("model.h5")
