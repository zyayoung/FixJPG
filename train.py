#%%
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
#%%
from utils import *
x_train, y_train = get_train_data((64,64), 9)

#%%
print(x_train.shape)
print(y_train.shape)

#%%
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)),
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    Conv2D(3,  (1,1), activation='relu'),
])
model.compile(loss="mse", optimizer=Adam(0.001))
model.fit(x_train, y_train, epochs=6, batch_size=128, validation_split=0.25)
model.compile(loss="mse", optimizer=SGD(0.0001))
model.fit(x_train, y_train, epochs=6, batch_size=128, validation_split=0.25)

model.save("model.h5")
