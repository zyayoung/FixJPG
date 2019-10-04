import sys
import cv2
import numpy as np
from model.resnet import resnet18sr
from model.srcnn import srcnn

model = srcnn()
model.load_weights("model_srcnn.h5")

im = cv2.imread("demo/xh.jpg")
# im = np.expand_dims(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),-1)
x = np.expand_dims(im, 0)/255
x = x.transpose(3,1,2,0)
y = model.predict(x)
for _ in range(1):
    y = model.predict(y)
y = y.transpose(3,1,2,0)
imy = np.uint8(np.clip(y[0], 0, 1)*255)
demo = np.vstack([im, imy, imy-im+127])

cv2.imwrite("demo/demo_srcnnh.jpg", demo)
cv2.imwrite("demo/y_srcnnh.jpg", imy)
