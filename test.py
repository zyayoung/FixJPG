import sys
import cv2
import numpy as np
from utils import get_raw_data
from model.resnet import resnet18sr
from model.srcnn import srcnn
from tqdm import tqdm

model = resnet18sr()
model.load_weights("model_resnet18sr.h5")

raw_psnr = []
fix_psnr = []
for xim, yim in tqdm(zip(*get_raw_data())):
    raw_psnr.append(cv2.PSNR(xim, yim))
    xim = np.expand_dims(xim, 0)/255
    y = model.predict(xim)
    for _ in range(0):
        y = model.predict(y)
    y = np.uint8(np.clip(y[0], 0, 1)*255)
    fix_psnr.append(cv2.PSNR(y, yim))
raw_psnr = np.array(raw_psnr)
print(raw_psnr.mean())
fix_psnr = np.array(fix_psnr)
print(fix_psnr.mean())
