from keras.utils import multi_gpu_model
from model.resnet import resnet18

from utils import get_train_data, extend_sym, PSNR

TRAIN_SIZE = 64

x_train, y_train = get_train_data((64, 64), 0)
x_train = extend_sym(x_train)
y_train = extend_sym(y_train)

model = resnet18((64, 64, 1))
model.load_weights("model.h5")

model_multi = multi_gpu_model(model, gpus=2)
model_multi.compile(loss="mse", optimizer="adam", metrics=[PSNR])

model_multi.fit(x_train, y_train, epochs=24, batch_size=256, validation_split=0.25)

model.save("model.h5")
