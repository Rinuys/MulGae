from keras import layers
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Inception 모듈 정의

def inception_module(x, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32):
    """
    # Arguments 
    x : 입력이미지
    o_1 : 1x1 convolution 연산 출력값의 채널 수 
    r_3 : 3x3 convolution 이전에 있는 1x1 convolution의 출력값 채널 수
    o_3 : 3x3 convolution 연산 출력값의 채널 수 
    r_5 : 5x5 convolution 이전에 있는 1x1 convolution의 출력값 채널 수 
    o_5 : 5x5 convolution 연산 출력값의 채널 수 
    pool: maxpooling 다음의 1x1 convolution의 출력값 채널 수
    
    # returns
    4 종류의 연산의 결과 값을 채널 방향으로 합친 결과 
    """
    
    x_1 = layers.Conv2D(o_1, 1, padding='same')(x)
    
    x_2 = layers.Conv2D(r_3, 1, padding='same')(x)
    x_2 = layers.Conv2D(o_3, 3, padding='same')(x_2)
    
    x_3 = layers.Conv2D(r_5, 1, padding='same')(x)
    x_3 = layers.Conv2D(o_5, 5, padding='same')(x_3)
    
    x_4 = layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    x_4 = layers.Conv2D(pool, 1, padding='same')(x_4)
    
    return layers.concatenate([x_1, x_2, x_3, x_4])

img_shape = (56, 56, 1)
def googLeNet(img_shape):
    input_ = layers.Input(shape=img_shape)
    x = layers.Conv2D(64, 3, strides=1, padding='same')(input_)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = inception_module(x, o_1=64, r_3=96, o_3=128, r_5=16, o_5=32, pool=32)
    x = inception_module(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = inception_module(x, o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)
    x = layers.AveragePooling2D(pool_size=(7, 7), strides=1)(x)
    x = layers.Dense(1024)(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(2350)(x)
    output = layers.Activation('softmax')(x)
    return Model(input_, output)

googlenet = googLeNet(img_shape)

# googlenet.summary()
print("전체 파라미터 수 : {}".format(sum([arr.flatten().shape[0] for arr in googlenet.get_weights()])))