from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
# include top을 False로 하면 마지막 부분에 fully connected layer가 사라짐. VGG16을 데이터에 맞게 커스터마이징 할 수 있음.
# model = VGG16()
# model = VGG19()
model.trainable=False # Trainable 0, None Trainable값으로 감 이 전이학습을 훈련시키지 않겠다는 뜻.
# 로직에 대해서 각각 가중치가 존재하는 것을 갱신하지 않겠다는 뜻. weight값 그대로 쓴다는 얘기다. ex) model predict
model.summary()

print(len(model.weights))
print(len(model.trainable_weights))
# 만약 shape가 다르다면, 이미지 사이즈를 늘리거나 줄여야함
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# ........................
# _________________________________________________________________
# flatten (Flatten)            (None, 25088)             0
# # _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================
# Total params: 143,667,240
# Trainable params: 143,667,240
# Non-trainable params: 0
# _________________________________________________________________

# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# 이미지넷 대회에서 VGG16이 우승했음.

# FC <- 용어정리