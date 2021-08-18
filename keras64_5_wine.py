import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.engine.base_layer_utils import create_keras_history
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Adam, Adadelta
# 실습
# cnn으로 변경
# 파라미터 변경해보고
# 노드 갯수, activation도 추가
# epochs = [123]
# learning_rate 추가 

# 나아아아중에 과제 : 레이어도 파라미터로 만들기- Dense, Conv <-

# 1. 데이터
dataset = load_wine()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

# 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('shape', x_train.shape)
print('shape', y_train.shape)
# '''
# 2. 모델
def build_model(drop=0.5, optimizer='adam', learning_rate=0.01, node=64, activation='relu'):
    inputs = Input(shape=(13,), name='input')
    x = Dense(node, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden3')(x)
    # x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden4')(x)
    # x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = optimizer(learning_rate)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [1, 8, 16, 32]
    optimizers = [RMSprop, Adam, Adadelta]
    dropout = [0.1] #, 0.2, 0.3]
    learning_rate = [0.01, 0.05, 0.001]
    node = [128, 64, 256]
    activation=['relu', 'sigmoid']
    return {"batch_size": batches, "optimizer": optimizers,
            'drop': dropout, 'learning_rate': learning_rate,
            'node': node, 'activation':activation}

hyperparameter = create_hyperparameter()
# print(hyperparameter)
# model = build_model()

model = KerasClassifier(build_fn=build_model, verbose=1, validation_split=0.2)

model = RandomizedSearchCV(model, hyperparameter, cv=5)
# mode2 = GridSearchCV(model, hyperparameter, cv=2)
# model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)
es = EarlyStopping(monitor='val_loss', mode='auto', patience=15)
model.fit(x_train, y_train, verbose=1, epochs=150, validation_split=0.2, callbacks=[es])

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc  = model.score(x_test, y_test)

print('최종 스코어 : ', acc)

# '''
# 최종 스코어 :  0.9814814925193787
# bset score :  0.6763333439826965