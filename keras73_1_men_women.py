import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, GlobalAveragePooling2D, Dense, Dropout, MaxPool2D, LSTM, SimpleRNN, Conv1D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2

# 실습
# men women 데이터로 모델링 구성하기
# 문제 1. 데이터 용량이 큼.

'''
train_datagen = ImageDataGenerator(
    rescale=1./120,
  )

start = time.time()
test_datagen = ImageDataGenerator(rescale=1./255)

xy = train_datagen.flow_from_directory(
    '../_data/men_women',
    target_size=(80, 80),
    batch_size=3309,
    class_mode='binary',
    shuffle=False
)
end = time.time() - start
# print('긁어오는 데 걸린 시간 :', end) 0.09275674819946289

print('xy : ',xy)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000022A82218550>
# print('xy[0] : ',xy[0]) # 라벨값이 나온다. y는 batch size.
# print('xy[0][0] : ',xy[0][0]) # x값
# print('xy[0][1] : ',xy[0][1]) # y값
# # print(xy_train[0][2]) # 없음
# print(xy[0][0].shape, xy[0][1].shape) # (3309, 250, 250, 3) (3309,)
np.save('./_save/_npy/k59_5_x.npy', arr=xy[0][0])
np.save('./_save/_npy/k59_5_x_predict.npy', arr=xy[1][0])
np.save('./_save/_npy/k59_5_y.npy', arr=xy[0][1])
'''

# 학습
# '''

x = np.load('./_save/_npy/k59_5_x.npy')
x_predict = np.load('./_save/_npy/k59_5_x_predict.npy')
y = np.load('./_save/_npy/k59_5_y.npy')

print('x shape', x.shape)
print('y shape', y.shape)
print('predict', x_predict)
print('predict shape', x_predict.shape)

# 실습 2 겸 과제
# 본인 사진으로 predict 하기. d:\data 안에 사진 넣고

# 모델 1

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=10)

incept = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(80, 80, 3))
model = Sequential()
model.add(incept)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))
# classification은 loss말고 acc가 더 중요한거같음.
model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(0.0005), metrics=['acc'])

learning_start = time.time()

es = EarlyStopping(monitor='val_acc', mode='max', patience=15)
# model.fit(x_train, y_train, epochs=50, validation_split=0.05, batch_size=8, callbacks=[es])
model.fit(x_train, y_train, epochs=50, validation_split=0.05, batch_size=126, callbacks=[es])
learning_end = (time.time() - learning_start)/60
print('학습 걸린 시간(분) : ', learning_end)

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])
print('asd', x_predict.shape)
y_predict = model.predict(x_predict)
y_value = np.around(y_predict)
print('y1', y_predict)
print('y1', y_value)
for i in range(0,5):
      print(f'{i} 번째 : {abs((1-y_predict[i])*100)}의 확률로 남자')
# '''

# Conv2d
# y_predict : [[0.6189552 ]
#  [0.03671281]
#  [0.84080505]
#  [0.5594377 ]
#  [0.45257118]]

# loss :  0.602618932723999
# acc :  0.6928499341011047

# inceptionResNetV2
# loss :  1.074310541152954
# acc :  0.7895267009735107

'''
Basic CNN
0 번째 : [75.04062]의 확률로 남자
1 번째 : [25.814837]의 확률로 남자
2 번째 : [24.355686]의 확률로 남자
3 번째 : [97.525116]의 확률로 남자
4 번째 : [62.31045]의 확률로 남자

inceptionResNetV2
0 번째 : [42.926163]의 확률로 남자
1 번째 : [16.606009]의 확률로 남자
2 번째 : [0.10549426]의 확률로 남자
3 번째 : [0.3805399]의 확률로 남자
4 번째 : [16.88264]의 확률로 남자
'''
# 0.7까지 올리기