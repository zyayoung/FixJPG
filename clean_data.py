#%%
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
#%%
from utils import *
x_train, y_train = get_full_data((256,256))

#%%
print(x_train.shape)
print(y_train.shape)

num = x_train.shape[0]
x_train_old = np.float32(x_train)/255.
x_train = np.concatenate([x_train, y_train])
x_train = np.float32(x_train)/255.
y_train = np.zeros((num*2, 2))
y_train[num:,0]=1
y_train[:num,1]=1

# enu = np.arange(2*num)
# np.random.shuffle(enu)
enu = np.arange(num*2)
enu = enu.reshape(2,-1).T.reshape(-1)
x_train = x_train[enu]
y_train = y_train[enu]

#%%
model = Sequential([
    Conv2D(64, (3,3), strides=2, activation='relu', input_shape=(256,256,3)),
    Conv2D(64, (3,3), strides=2, activation='relu'),
    Conv2D(64, (3,3), strides=2, activation='relu'),
    Conv2D(64, (3,3), strides=2, activation='relu'),
    Flatten(),
    Dense(2, activation='softmax')
])
model.summary()
model.compile(loss=categorical_crossentropy, optimizer=Adam(0.001), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=6, shuffle=False)
model.compile(loss=categorical_crossentropy, optimizer=Adam(0.0001), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=6)

pred = model.predict(x_train_old)[:,0]
names = get_name_data()
for arg in np.argsort(pred)[-50:]:
    print(names[arg])
    os.remove('x/'+names[arg])
