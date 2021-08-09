import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.7, shuffle=True, random_state=22)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# case1 연산이 너무 적음 5,4,3,2,1
# case2 노드가 많다가 갑자기 줄어듬. 400, 243, 3, 1
# case3. 1000, 5883, 840, 1233, 1102, 8335 통상적으로 역삼각형 형태가 가장 많음.
# 한 레이어당 지나치게 많은 노드의 수를 배정할 경우 과적합이 일어난다.
model = Sequential()
model.add(Dense(128, input_shape=(10, ), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# model.summary()
# batch size와 epoch를 둘다 늘릴 경우, 일반화된 데이터의 과적합으로 인해 val_loss가 하락하는 것

#2. 모델구성
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=9, train_size=0.7)

#3. 컴파일, 훈련

optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto')
# cp = ModelCheckpoint(monitor='val_accuracy', mode='max', filepath='./_save/ModelCheckPoint/keras48_2_MCP.hdf', save_best_only=True)

es = EarlyStopping(mode='auto', monitor='val_loss', patience=15)
start_time = time.time()
model.fit(x_train, y_train, epochs=80, validation_split=0.1, batch_size=45, callbacks=[es, reduce_lr])
end_time = time.time() - start_time
# model.save('./_save/ModelCheckPoint/keras48_2_model.h5')

# model =load_model('./_save/ModelCheckPoint/keras48_2_model.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_2_MCP.hdf')

#4. 평가, 예측
# mse, R2
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# 과제 2
# r2 score :  0.6153617787632967
# load
# r2 score :  0.6042646163300688

# lr reduce 사용 후 - 오히려 learning rate를 줄이니까 r2 score가 감소
# r2 score :  0.5917474059098785