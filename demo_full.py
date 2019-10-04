import sys
import cv2
import numpy as np
from model.resnet import resnet18sr
from model.srcnn import srcnn
from prepare_data import compress_image

model = resnet18sr()
model.load_weights("model_resnet18sr.h5")

compress_image("h.jpg", "hx.jpg", 4)

im = cv2.imread("hx.jpg")
gt = cv2.imread("h.jpg")
# im = np.expand_dims(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),-1)
x = np.expand_dims(im, 0)/255
x = x.transpose(3,1,2,0)
y = model.predict(x)
for _ in range(1):
    y = model.predict(y)
y = y.transpose(3,1,2,0)
imy = np.uint8(np.clip(y[0], 0, 1)*255)

cv2.putText(im, "psnr: {:.2f}dB".format(cv2.PSNR(gt, im)), (30, im.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
im[3:75,-127:-3] = cv2.resize(im[209:245,88:150], (124, 72))
cv2.rectangle(im, (88, 209), (150, 245), (127,63,63), 2)
cv2.rectangle(im, (im.shape[1]-124-3, 3), (im.shape[1]-3, 72+3), (127,63,63), 3)

cv2.putText(imy, "psnr: {:.2f}dB".format(cv2.PSNR(gt, imy)), (30, im.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
imy[3:75,-127:-3] = cv2.resize(imy[209:245,88:150], (124, 72))
cv2.rectangle(imy, (88, 209), (150, 245), (127,63,63), 2)
cv2.rectangle(imy, (im.shape[1]-124-3, 3), (im.shape[1]-3, 72+3), (127,63,63), 3)

gt[3:75,-127:-3] = cv2.resize(gt[209:245,88:150], (124, 72))
cv2.rectangle(gt, (88, 209), (150, 245), (127,63,63), 2)
cv2.rectangle(gt, (im.shape[1]-124-3, 3), (im.shape[1]-3, 72+3), (127,63,63), 3)

demo = np.hstack([gt, im, imy])

cv2.imwrite("demo_resnet18srh.jpg", demo)
