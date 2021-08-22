# 실습
# 약속 잘 지키기 왕
# 말과 사람 데이터셋 완성
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D, MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.core import Flatten
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2

# '''
train_datagen = ImageDataGenerator(rescale=1./255,)

xy = train_datagen.flow_from_directory(
    '../_data/horse-or-human',
    target_size=(100, 100),
    batch_size=1027,
    class_mode='sparse',
    shuffle=True
)

np.save('./_save/_npy/k59_7_x.npy', arr=xy[0][0])
np.save('./_save/_npy/k59_7_y.npy', arr=xy[0][1])
# '''

# 학습
# '''
x = np.load('./_save/_npy/k59_7_x.npy')
y = np.load('./_save/_npy/k59_7_y.npy')

x_train, x_test, y_train, x_test = train_test_split(x, y, test_size=0.7, shuffle=True, random_state=10)

model = Sequential()
incept = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
model.add(incept)
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=15, mode='min')
model.fit(x_train, y_train, validation_split=0.1, epochs=150, callbacks=[es])

result = model.evaluate(x_train, y_train)
print('loss :', result[0])
print('acc :', result[1])
# '''
# basic CNN
# size 250, 250
# loss : 0.056782789528369904
# acc : 0.9935064911842346

# size 100, 100
# loss : 0.001572958892211318
# acc : 1.0

# InceptionResNetV2
# loss : 5.029321982874535e-06
# acc : 1.0