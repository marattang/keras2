from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
# include top을 False로 하면 마지막 부분에 fully connected layer가 사라짐. VGG16을 데이터에 맞게 커스터마이징 할 수 있음.
# model = VGG16()
# model = VGG19()

vgg16.trainable=True

vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))

# model.trainable=False # 훈련을 동결한다 여기선 사용하면 안됨.

model.summary()

# 이거는 좀더 봐야할 거 같음
print(len(model.weights))           # 26 -> 30 maxpooling 제외하고 13개 레이어, 마지막 두 개의 dense레이어를 합해서 15개의 레이어의 각각 기울기, 절편을 구해야 하기 때문에
# 30이 된다.
print(len(model.trainable_weights)) # 0 -> 4 

###################2번 파일에서 아래만 추가

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable)for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

print(results)