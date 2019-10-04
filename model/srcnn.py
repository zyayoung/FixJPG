from keras.layers import Conv2D, Dense, Input
from keras.models import Model

def srcnn(input_shape=(None, None, 1)):
    x = Input(input_shape)
    y = Conv2D(64, (9,9), padding='same', activation='relu')(x)
    y = Conv2D(32, (1,1), padding='same', activation='relu')(y)
    y = Conv2D(1, (5,5), padding='same')(y)
    return Model(x, y)
