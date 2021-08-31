# 실습
# cifar10과 cifar100으로 모델 만들기
# trainable=True, False
# FC로 만든 것과 Average Pooling로 만든거 비교

# 결과치
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100, cifar10
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

# 전처리
dataset = [cifar10, cifar100]

# 한번에 돌릴려면 하나씩 줘야하는데
def model_train(dataset):
    option = False
    for i in range(2):
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        x_train = x_train.reshape(50000, 32, 32, 3)
        x_test = x_test.reshape(10000, 32, 32, 3)
        resnet = ResNet101(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        # include top을 False로 하면 마지막 부분에 fully connected layer가 사라짐. VGG16을 데이터에 맞게 커스터마이징 할 수 있음.
        model1 = Sequential()
        model2 = Sequential()
        resnet.trainable=option
        # vgg19.trainable=True
        # FC/
        model1.add(resnet)
        model1.add(Flatten())
        model1.add(Dense(100, activation='relu'))
        model1.add(Dense(100, activation='softmax'))
        # GAP
        model2.add(resnet)
        model2.add(GlobalAveragePooling2D())
        model2.add(Dense(50, activation='relu'))
        model2.add(Dense(100, activation='softmax'))

        model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        es = EarlyStopping(mode='min', monitor='val_loss', patience=6)
        cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_8_MCP.hdf', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=3, mode='auto')
        start = time.time()

        tf.debugging.set_log_device_placement(True)
        # with tf.device('/GPU:1'):
        model1.fit(x_train, y_train, epochs=500, batch_size=512, validation_split=0.05, callbacks=[es, reduce_lr], verbose=3)
        model2.fit(x_train, y_train, epochs=500, batch_size=512, validation_split=0.05, callbacks=[es, reduce_lr], verbose=3)
        end = time.time() - start

        print("걸린시간 : ", end)

        loss = model1.evaluate(x_test, y_test)
        print(f'==========={dataset}')
        print(f'FC/trainable{option}')
        print('loss : ', loss[0], 'accuracy : ', loss[1])

        loss = model2.evaluate(x_test, y_test)
        print(f'GAP/trainable{option}')
        print('loss : ', loss[0], 'accuracy : ', loss[1])

        if option == True:
            option = False
        else :
            option = True
        # 이거는 좀더 봐야할 거 같음

model_train(cifar10)
model_train(cifar100)

# 1. cifar 10
# trainable = True, FC : loss = ?, acc = ?
# trainable = True, GAP : loss = ? acc = ?
# trainable = False, FC : loss = ? , acc = ?
# trainable = False, GAP : loss = ? , acc = ?

# 2. cifar 100
# trainable = True, FC : loss = ?, acc = ?
# trainable = True, GAP :  loss : 4.605202674865723 acc : 0.009999999776482582
# trainable = False, FC : loss = ? , acc = ?
# trainable = False, GAP : loss :  3.0389254093170166 , accuracy :  0.27549999952316284

'''
cifar 10
FC/trainable : False
loss :  1.2076201438903809 accuracy :  0.6093999743461609
GAP/trainable : False
loss :  1.1608192920684814 accuracy :  0.6107000112533569
FC/trainable : True
loss :  0.7373404502868652 accuracy :  0.8233000040054321
GAP/trainable : True
loss :  0.9726871252059937 accuracy :  0.823199987411499
 
cifar100
FC/trainable : False
loss :  2.7700016498565674 accuracy :  0.3483999967575073
GAP/trainable : False
loss :  2.781135320663452 accuracy :  0.335099995136261
FC/trainable : True
loss :  3.200453042984009 accuracy :  0.5105999708175659
GAP/trainable : True
loss :  3.0223426818847656 accuracy :  0.5260999798774719
'''