import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

print(model.weights) # <- 초기값 weight, parameter별(bias param 제외) 초기 weight 값을 출력
print('==================================')
# print(model.trainable_weights)
print('==================================')
print(len(model.weights))
print(len(model.trainable_weights))
# model.summary()
# layer마다 weight, bias가 하나씩 있기 때문에 3(w+b) = 3(1+1) => 총 6개가 된다.
'''
VGG16같은 모델 pretrained learning 사전모델 transfer model
Total params: 17
Trainable params: 17 -> ? 전이학습할때 값이 바뀐다.
Non-trainable params: 0 -> ?
'''
# layer에서 kerenel = weight
# kernel initializer ? 
# kernel regulizer ? -> L1, L2로 나뉜다. 라소와 릿치?
# bias는 default = 0