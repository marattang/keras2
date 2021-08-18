import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.engine.base_layer_utils import create_keras_history
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 전처리
# 데이터의 내용물과 순서가 바뀌면 안된다.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28 * 28).astype('float32')/255
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 28 * 28).astype('float32')/255


# 2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [100, 200, 300, 400, 500]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1] #, 0.2, 0.3]
    return {"batch_size": batches, "optimizer": optimizers,
            'drop': dropout}

hyperparameter = create_hyperparameter()
# print(hyperparameter)
# model = build_model()

model = KerasClassifier(build_fn=build_model, verbose=1, validation_split=0.2)

model = RandomizedSearchCV(model, hyperparameter, cv=5)
# mode2 = GridSearchCV(model, hyperparameter, cv=2)

# model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)
model.fit(x_train, y_train, verbose=1, epochs=30)

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc  = model.score(x_test, y_test)

print('최종 스코어 : ', acc)

