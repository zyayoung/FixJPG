#%%
from keras.layers import Conv2D, Flatten, Dense, Input, add, BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
from keras.utils import multi_gpu_model
#%%
from utils import *
x_train, y_train = get_train_data((64,64), 0)

import os
os.environ['CUDA_VISIABLE_DEVICES'] = '-1'

#%%
print(x_train.shape)
print(y_train.shape)

def residual_unit_b(x):
    y = Conv2D(64, (1,1), padding='same')(x)
    # y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(64, (3,3), padding='same')(y)
    # y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(256, (1,1), padding='same')(y)
    # y = BatchNormalization()(y)

    out = add([x, y])
    return Activation('relu')(out)

def residual_unit(x):
    y = Conv2D(64, (3,3), padding='same')(x)
    # y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(64, (3,3), padding='same')(y)
    # y = BatchNormalization()(y)

    out = add([x, y])
    return Activation('relu')(out)

#%%
x = Input((64,64,1))
y = Conv2D(64, (3,3), activation='relu', padding='same')(x)
y = residual_unit(y)
y = residual_unit(y)
y = residual_unit(y)
y = residual_unit(y)
y = residual_unit(y)
y = residual_unit(y)
y = residual_unit(y)
y = residual_unit(y)
y = Conv2D(1, (1,1), padding='same')(y)
y = add([x, y])
model = Model(x, y)

model_multi = multi_gpu_model(model, gpus=2)
model_multi.compile(loss="mse", optimizer=Adam(0.001))
x_train = np.concatenate([x_train.transpose(0,2,1,3), x_train])
y_train = np.concatenate([y_train.transpose(0,2,1,3), y_train])
x_train = np.concatenate([x_train[:,:,::-1], x_train])
y_train = np.concatenate([y_train[:,:,::-1], y_train])
x_train = np.concatenate([x_train[:,::-1,:], x_train])
y_train = np.concatenate([y_train[:,::-1,:], y_train])
model_multi.fit(x_train, y_train, epochs=24, batch_size=256, validation_split=0.25)

model.save("model.h5")
