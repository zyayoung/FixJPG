#%%
from keras.layers import Conv2D, Flatten, Dense
from keras.models import load_model, Sequential
from keras.optimizers import Adam
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

#%%
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(None,None,3)),
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
model.load_weights("model.h5")

#%%
im = cv2.imread("test.jpg")
# im = np.expand_dims(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),-1)
im = np.pad(im, ((9,9),(9,9),(0,0),),'edge')
imy = model.predict(np.expand_dims(im, 0)/255)[0]
imy = np.uint8(np.clip(imy*255, 0, 255))
# demo = np.vstack([im, imy])
# plt.imshow(imy)
#%%
# cv2.imwrite("demo.jpg", demo)
cv2.imwrite("y.jpg", imy)

#%%
# im.shape

#%%
