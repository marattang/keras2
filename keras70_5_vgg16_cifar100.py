from tensorflow.keras.applications import VGG16, VGG19
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM, GlobalAveragePooling2D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.python.keras.layers.core import Dropout
import numpy as np
import time
import tensorflow as tf
# 실습 cifar10 완성
# 동결하고, 안하고
# FC를 모뗼로 하꼬 average pooling으로 하고

# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 전처리
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# include top을 False로 하면 마지막 부분에 fully connected layer가 사라짐. VGG16을 데이터에 맞게 커스터마이징 할 수 있음.
# model = VGG16()
# model = VGG19()

# vgg16.trainable=False

vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='softmax'))

# model.trainable=False # 훈련을 동결한다 여기선 사용하면 안됨.

model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(mode='min', monitor='val_loss', patience=5)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_8_MCP.hdf', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto')
start = time.time()

tf.debugging.set_log_device_placement(True)
with tf.device('/GPU:1'):
    model.fit(x_train, y_train, epochs=500, batch_size=512, validation_split=0.05, callbacks=[es, reduce_lr])
end = time.time() - start

print("걸린시간 : ", end)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# 이거는 좀더 봐야할 거 같음
print(len(model.weights))           # 26 -> 30
print(len(model.trainable_weights)) # 0 -> 4




