import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.engine.base_layer_utils import create_keras_history
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.datasets import load_diabetes
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
dataset = load_diabetes()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

# 전처리

print('shape', x_train.shape)
print('shape', y_train.shape)
# '''
# 2. 모델
def build_model(drop=0.5, optimizer='adam', learning_rate=0.01, node=64, activation='relu'):
    inputs = Input(shape=(10,), name='input')
    x = Dense(node, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node/2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node/2, activation=activation, name='hidden3')(x)
    x = Dense(node/4, activation=activation, name='hidden4')(x)
    x = Dense(node/4, activation=activation, name='hidden5')(x)
    x = Dense(node/8, activation=activation, name='hidden6')(x)
    optimizer = optimizer(learning_rate)
    outputs = Dense(1, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

def create_hyperparameter():
    batches = [1, 8, 16, 32]
    optimizers = [RMSprop, Adam, Adadelta]
    dropout = [0.1] #, 0.2, 0.3]
    learning_rate = [0.01, 0.05, 0.001]
    node = [128, 64]
    activation=['relu', 'sigmoid']
    return {"batch_size": batches, "optimizer": optimizers,
            'drop': dropout, 'learning_rate': learning_rate,
            'node': node, 'activation':activation}

hyperparameter = create_hyperparameter()
# print(hyperparameter)
# model = build_model()

model = KerasRegressor(build_fn=build_model, verbose=1, validation_split=0.2)

model = RandomizedSearchCV(model, hyperparameter, cv=2)
# '''
# mode2 = GridSearchCV(model, hyperparameter, cv=2)
# model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)
es = EarlyStopping(monitor='val_loss', mode='auto', patience=5)
model.fit(x_train, y_train, verbose=1, epochs=150, validation_split=0.2, callbacks=[es])

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
r2  = model.score(x_test, y_test)

print('최종 스코어 : ', r2)

# 최종 스코어 :  -0.0032052318565547466
'''
{'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>,
 'node': 128,
  'learning_rate': 0.05,
'drop': 0.1, 'batch_size': 1, 'activation': 'sigmoid'}
<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x0000022DAF618B50>
'''
# 