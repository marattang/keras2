import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense    
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

#1. 데이터       
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,6,7,8,9,10,11,12])

#2. 모델

model = Sequential()  
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
# optimizer = Adam(lr=0.1)
# optimizer = Adagrad(lr=0.1)
# optimizer = Adadelta(lr=0.1)
# optimizer = Adamax(lr=0.1)
# optimizer = RMSprop(lr=0.1)
# optimizer = SGD(lr=0.1)
optimizer = Nadam(lr=0.1)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)
# loss :  0.012453261762857437 결과물 :  [[11.246211]]

# Adam 0.1, 0.01, 0.001,
# loss :  22.829082489013672 결과물 :  [[8.180467]]
# loss :  0.4115850329399109 결과물 :  [[14.397665]]
# loss :  0.25253114104270935 결과물 :  [[13.789201]]

# Adagrad 0.1, 0.01, 0.001,
# loss :  14.50390625 결과물 :  [[7.3950005]]
# loss :  0.22908830642700195 결과물 :  [[13.711984]]
# loss :  0.22799105942249298 결과물 :  [[13.718866]]

# Adadelta 0.1, 0.01, 0.001,
# loss :  0.22159788012504578 결과물 :  [[13.521163]]
# loss :  0.26683109998703003 결과물 :  [[13.796536]]
# loss :  6.579311370849609 결과물 :  [[9.517725]]

# Adamax 0.1, 0.01, 0.001,
# loss :  0.30262118577957153 결과물 :  [[13.086632]]
# loss :  0.25274738669395447 결과물 :  [[13.164889]]
# loss :  0.22096964716911316 결과물 :  [[13.569759]]

# RMSprop 0.1, 0.01, 0.001,
# loss :  123466512.0 결과물 :  [[-8187.761]]
# loss :  39.656471252441406 결과물 :  [[1.7676103]]
# loss :  0.9135068655014038 결과물 :  [[15.055128]]

# SGD
# loss :  nan 결과물 :  [[nan]]
# loss :  nan 결과물 :  [[nan]]
# loss :  0.22361163794994354 결과물 :  [[13.632466]]

# Nadam
# loss :  0.26457372307777405 결과물 :  [[13.912681]]
# loss :  0.2523295283317566 결과물 :  [[13.159658]]
# loss :  0.6196816563606262 결과물 :  [[12.504545]]