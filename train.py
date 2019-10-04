from keras.utils import multi_gpu_model
from model.resnet import resnet18sr
from model.srcnn import srcnn

from utils import get_train_data, extend_sym, PSNR

TRAIN_SIZE = 64
NUM_GPU = 1

x_train, y_train = get_train_data((64, 64), 0)
x_train = extend_sym(x_train)
y_train = extend_sym(y_train)

model = srcnn((64, 64, 1))
# model.load_weights("model.h5")

model_multi = model if NUM_GPU == 1 else multi_gpu_model(model, gpus=NUM_GPU)
model_multi.compile(loss="mse", optimizer="adam", metrics=[PSNR])

model_multi.fit(x_train, y_train, epochs=24, batch_size=256, validation_split=0.25)

model.save("model_srcnn.h5")
