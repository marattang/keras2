import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling1D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.python.keras.layers.core import Dropout
import time
# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,) 흑백데이터이기 때문에 3차원
# print(x_test.shape, y_test.shape)   (10000, 28, 28) (10000,)

# 전처리
# x_train = x_train.reshape(60000, 28 * 28)
# x_test = x_test.reshape(10000, 28 * 28)

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
print('y_shape : ', y_train.shape)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np.c_[y_train.toarray()]
y_test = encoder.fit_transform(y_test)
y_test = np.c_[y_test.toarray()]

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. 모델링
model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# DNN, CNN 비교
# DNN 구해서 CNN 비교
# 3시 50분까지

# 3. 컴파일, 훈련       metrics['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(mode='min', monitor='val_loss', patience=15)
cp = ModelCheckpoint(monitor='val_accuracy', mode='max', filepath='./_save/ModelCheckPoint/keras48_6_MCP.hdf', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.05, callbacks=[es, reduce_lr])
end = time.time() - start
print('걸린시간 : ', end)

# model.save('./_save/ModelCheckPoint/keras48_6_model.h5')
# model =load_model('./_save/ModelCheckPoint/keras48_6_model.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_6_MCP.hdf')


# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# 0.9829999804496765
# learning rate reduce 사용 후
# accuracy :  0.98580002784729