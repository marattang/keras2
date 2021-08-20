# pre-trained model

from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7

# model = Xception()
# model = ResNet50()
# model = ResNet50V2()
# model = ResNet101()
# model = ResNet101V2()
# model = ResNet152()
# model = ResNet152V2()
model = DenseNet121()
model = DenseNet169()
model = DenseNet201()
model.trainable =False

model.summary()

print('전체 가중치 갯수 : ',len(model.weights))
print('훈련가능 가중치 갯수 : ',len(model.trainable_weights))

# VGG16
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# 30
# 30