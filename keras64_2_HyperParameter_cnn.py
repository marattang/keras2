import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.engine.base_layer_utils import create_keras_history
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import RMSprop, Adam, Adadelta
import random as rd
import warnings

warnings.filterwarnings('ignore')
# 실습
# cnn으로 변경
# 파라미터 변경해보고
# 노드 갯수, activation도 추가
# epochs = [123]
# learning_rate 추가 

# 나아아아중에 과제 : 레이어도 파라미터로 만들기- Dense, Conv <-

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 전처리
# 데이터의 내용물과 순서가 바뀌면 안된다.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

# 2. 모델
# '''
def build_model(drop=0.5, optimizer=Adam, learning_rate=0.01, activation='relu', node=[32, 64, 128, 256]):
    inputs = Input(shape=(28, 28, 1), name='input')
    x = Conv2D(node, kernel_size=(2,2), activation=activation)(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(node, kernel_size=(2,2), activation=activation)(x)
    x = Conv2D(node, kernel_size=(2,2), activation=activation)(x)
    x = Flatten()(x)
    x = Dense(node, activation=activation)(x)
    x = Dense(node, activation=activation)(x)
    print(type(x)) #<class 'tensorflow.python.keras.engine.keras_tensor.KerasTensor'>
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    optimizer = optimizer(learning_rate)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def ran_int(node):
    print('node :', node)
    print('node :', type(node))
    intvalue = rd.choice(node)
    print('type:', intvalue)
    return rd.choice(node)
'''
def cnn_layer(node=[128, 32], kernel_size) :

    for i in node:
        Conv2D(node, kernel_size=kernel_size, activation='relu')
    return Conv2D    
'''
def create_hyperparameter():
    batches = [1000, 2000, 3000, 4000, 5000]
    optimizers = [RMSprop, Adam, Adadelta]
    dropout = [0.1] #, 0.2, 0.3]
    learning_rate = [0.01, 0.05, 0.001]
    node = [8, 32, 128, 64, 256]
    activation=['relu', 'sigmoid']
    return {"batch_size": batches, "optimizer": optimizers,
            'drop': dropout, "learning_rate":learning_rate,
            'activation': activation, 'node':node}

hyperparameter = create_hyperparameter()
# print(hyperparameter)
# model = build_model()

model = KerasClassifier(build_fn=build_model, verbose=1, validation_split=0.2)

model = RandomizedSearchCV(model, hyperparameter, cv=5)
# mode2 = GridSearchCV(model, hyperparameter, cv=2)

# model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
model.fit(x_train, y_train, verbose=1, epochs=123, callbacks=[es])
# 어기적~ 어기적~
print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc  = model.score(x_test, y_test)

print('최종 스코어 : ', acc)

