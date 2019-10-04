#%%
from keras.layers import Conv2D, Flatten, Dense, Input, add, BatchNormalization, Activation
from keras.models import load_model, Sequential, Model
from keras.optimizers import Adam
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

#%%

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
x = Input((None, None, 1))
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
model.load_weights("model.h5")

#%%
im = cv2.imread("test.jpg")
# im = np.expand_dims(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),-1)
# im = np.pad(im, ((9,9),(9,9),(0,0),),'edge')
x = np.expand_dims(im, 0)/255
x = x.transpose(3,1,2,0)
y = model.predict(x)
y = y.transpose(3,1,2,0)
# y = model.predict(y)
# y = model.predict(y)
# y = model.predict(y)
# y = model.predict(y)
imy = y[0]
imy = np.uint8(np.clip(imy*255, 0, 255))
# demo = np.vstack([im, imy])
# plt.imshow(imy)
#%%
# cv2.imwrite("demo.jpg", demo)
cv2.imwrite("y.jpg", imy)

#%%
# im.shape

#%%
