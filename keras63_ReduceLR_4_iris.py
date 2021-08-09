import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, LSTM, Dropout
from sklearn.metrics import r2_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.optimizers import Adam

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target


# 데이터 자르기 전에 One-Hot-Encoding (150,) -> (150, 3, 1)
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

# [0, 1, 2, 1 ->
# [[1, 0, 0]
# [0, 1, 0]
# [0, 0, 1]
# [0, 1, 0]] (4,) -> (4, 3) 원핫 인코딩 후 라벨의 종류만큼 컬럼 열이 생겨나는 것이다.
y = to_categorical(y)
print(y[:5])
print(y.shape)
# print(x.shape, y.shape) # (150,4) (150,)

# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=9, train_size=0.7)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)
print(x_test.shape)
# x_train = x_train.reshape(105, 4, 1)
# x_test = x_test.reshape(45, 4, 1)
#2. 모델구성
# model = Sequential()
# model.add(LSTM(64, input_shape=(4, 1)))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(3, activation='softmax'))

model = Sequential()
model.add(Dense(128, input_shape=(4, ), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print(x_train.shape)
print(y_train.shape)
# cp = ModelCheckpoint(filepath='./_save/ModelCheckPoint/keras48_4_MCP.hdf', mode='auto', monitor='val_loss', save_best_only=True)
es = EarlyStopping(mode='auto', monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto')
optimizer = Adam(lr=0.001)
model.fit(x_train, y_train, epochs=200, validation_split=0.1, batch_size=1, callbacks=[es, reduce_lr])

# model.save('./_save/ModelCheckPoint/keras48_4_model.h5')
# model =load_model('./_save/ModelCheckPoint/keras48_4_model.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_4_MCP.hdf')

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


# DNN
# accuracy :  0.9111111164093018

# lr reduce 사용 후 
# accuracy :  1.0