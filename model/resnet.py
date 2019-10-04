from keras.layers import Conv2D, Flatten, Dense, Input, add, BatchNormalization, Activation
from keras.models import Model
import cv2
import numpy as np


def residual_unit_bottleneck(x):
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

def resnet18(input_shape=(None, None, 1)):
    x = Input(input_shape)
    y = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    for _ in range(8):
        y = residual_unit(y)
    y = Conv2D(1, (1,1), padding='same')(y)
    y = add([x, y])
    return Model(x, y)
