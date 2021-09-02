# overfit 극복~~
# 1. 전체 훈련 데이터를 늘린다. 많이 많이 train data가 많으면 많을 수록 오버핏이 준다. -> 실질적으로 불가능한 경우가 많다.
#  - 증폭시킬려고 해도 비슷한 양으로 증폭되기 때문에 한계가 있다.
# 2. Normalization : 정규화
#  - Regulization, Standardzation이랑 헷갈리지 말기. layer값에서 다음 값으로 전달해 줄 때 activation으로 값을 감싸서 다음 layer로 전달해주게 되는데
#  - 그 값 자체도 Normalize 하지 않다는 얘기다. layer별로도 Normalize해주는 게 어떠냐는 얘기?
# 3. dropout
#  - 전체적으로 연결되어있는 레이어의 구성을 Fully Connected layer라고 하는데, 
# 완벽한 모델 구성import numpy as np

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus :
#     try :
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

#     except RuntimeError as e:
#         print(e)
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100, mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool2D, GlobalAveragePooling2D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.distribute.distribute_lib import Strategy
from tensorflow.python.keras.layers.core import Dropout
import numpy as np
import time
import matplotlib.pyplot as plt

# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 전처리
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 28, 28, 1)/255.


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# # strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
#         # tf.distribute.HierarchicalCopyAllReduce() # 375 => size / batch size? 205
#         tf.distribute.ReductionToOneDevice() 제일 빠름 204
            
#     )
# strategy = tf.distribute.MirroredStrategy(
#     # devices=['/gpu:0'] 361
#     # devices=['/gpu:1'] 1보다 0이 빨랐음 388
#     # devices=['/cpu', '/gpu:0'] # cpu gpu 둘다 사용한다. 
#     devices=['/gpu:0', '/gpu:1']  # gpu 같이 쓸려면 위에 HierarchicalCopyAllReduce 랑 ReductionToOneDevice 쓰면 된다. 에러
# )

# strategy = tf.distribute.experimental.CentralStorageStrategy()  204
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy( # 209
    tf.distribute.experimental.CollectiveCommunication.RING      # 205
    # tf.distribute.experimental.CollectiveCommunication.NCCL   # 204
    # tf.distribute.experimental.CollectiveCommunication.AUTO   # 205
# 위 세 개 중에 아무거나 써도 되는데, reload strategy 이후로 그렇게 성능차이 안남. 
) #

# tnesorflow org 들어가면 분산형 학습이라고 있음. 심심할 때 보기
# 현재 버전에서 실행이 안되는 부분이 몇 개 있음.  device를 먹혀주던가 cross device option을 설정해줘야지 제대로 먹힌다.
# 분산처리할 때는 배치 사이즈가 클 수록 좋다. 두개가 같이 돌기 때문에
with strategy.scope():
    model = Sequential()
    model.add(Conv2D(filters=128, activation='relu', kernel_size=(2,2), padding='valid',  input_shape=(28, 28, 1)))
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
    model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련       metrics['acc']
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, verbose=1)
# hist = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.1, callbacks=[es], verbose=1)
end = time.time() - start

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()

print("걸린시간 : ", end)
# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])

# CNN
# acc :  0.4796999990940094

# DNN
# acc :  0.24869999289512634