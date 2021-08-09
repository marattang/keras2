# overfit 극복~~
# 1. 전체 훈련 데이터를 늘린다. 많이 많이 train data가 많으면 많을 수록 오버핏이 준다. -> 실질적으로 불가능한 경우가 많다.
#  - 증폭시킬려고 해도 비슷한 양으로 증폭되기 때문에 한계가 있다.
# 2. Normalization : 정규화
#  - Regulization, Standardzation이랑 헷갈리지 말기. layer값에서 다음 값으로 전달해 줄 때 activation으로 값을 감싸서 다음 layer로 전달해주게 되는데
#  - 그 값 자체도 Normalize 하지 않다는 얘기다. layer별로도 Normalize해주는 게 어떠냐는 얘기?
# 3. dropout
#  - 전체적으로 연결되어있는 레이어의 구성을 Fully Connected layer라고 하는데, 
# 완벽한 모델 구성import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool2D, GlobalAveragePooling1D, LSTM, GlobalAveragePooling2D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.python.keras.layers.core import Dropout
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


# 전처리
x_train = x_train.reshape(50000, 32 * 32 * 3)
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 32 * 32 * 3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 32, 32, 3)

# x_train = x_train.reshape(50000, 32, 32, 3)
# 데이터의 내용물과 순서가 바뀌면 안된다.
# x_test = x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
print('y_shape : ', y_train.shape)
y_train = y_train.reshape(50000, 1)
y_test = y_test.reshape(10000, 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np.c_[y_train.toarray()]
y_test = encoder.fit_transform(y_test)
y_test = np.c_[y_test.toarray()]

# CNN
model = Sequential()
model.add(Conv2D(filters=128, activation='relu', kernel_size=(2,2), padding='valid',  input_shape=(32, 32, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(128, (2,2), activation='relu', padding='valid'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(GlobalAveragePooling2D())      
model.add(Dense(100, activation='softmax'))


# 3. 컴파일, 훈련       metrics['acc']
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

es = EarlyStopping(mode='auto', monitor='val_loss', patience=15)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_9_MCP.hdf', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5, )
# val_loss가 떨어지다가 5번 갱신이 없으면 learnign rate가 50%감축된다.
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.05, verbose=1, callbacks=[es, cp, reduce_lr])
# model.save('./_save/ModelCheckPoint/keras48_9_model.h5')
# model =load_model('./_save/ModelCheckPoint/keras48_9_model.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_9_MCP.hdf')

end = time.time() - start

print("걸린시간 : ", end)
# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])

# CNN
# acc :  0.4796999990940094

# learning rate reduce 사용 후 
# acc :  0.4846999943256378