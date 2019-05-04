from keras import layers
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD
from datetime import datetime
import keras.backend.tensorflow_backend as Back
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
    
    x_1 = layers.Conv2D(o_1, 1, padding='same', activation='relu')(x)
    
    x_2 = layers.Conv2D(r_3, 1, padding='same', activation='relu')(x)
    x_2 = layers.Conv2D(o_3, 3, padding='same', activation='relu')(x_2)
    
    x_3 = layers.Conv2D(r_5, 1, padding='same', activation='relu')(x)
    x_3 = layers.Conv2D(o_5, 5, padding='same', activation='relu')(x_3)
    
    x_4 = layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    x_4 = layers.Conv2D(pool, 1, padding='same', activation='relu')(x_4)
    
    return layers.concatenate([x_1, x_2, x_3, x_4])

img_shape = (56, 56, 1)
def googLeNet(img_shape):
    with Back.tf.device('/gpu:0'):
        input_ = layers.Input(shape=img_shape)
        x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(input_)
        x = layers.Conv2D(192, 3, strides=1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        x = inception_module(x, o_1=64, r_3=96, o_3=128, r_5=16, o_5=32, pool=32)
        x = inception_module(x, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        x = inception_module(x, o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)
        x = layers.AveragePooling2D(pool_size=(7, 7), strides=1)(x)
        x = layers.Conv2D(128, 1, strides=1, padding='same', activation='relu')(x)
        x = layers.Dense(1024)(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Dense(2350)(x)
        x = layers.Reshape((2350,), input_shape=(1,1,2350))(x)
        output = layers.Activation('softmax')(x)
    return Model(input_, output)

googlenet = googLeNet(img_shape)

# googlenet.summary()
print("전체 파라미터 수 : {}".format(sum([arr.flatten().shape[0] for arr in googlenet.get_weights()])))

with Back.tf.device('/gpu:0'):
    googlenet.compile(optimizer=SGD(lr=0.01,decay=1e-6, momentum=0.9), 
    loss='categorical_crossentropy', metrics=["accuracy"])
#googlenet.summary()
path = 'D:\PHD08\phd08-conversion-master\phd08_npy_results\phd08_'
def next_load_data(path, train, index):
    #DataSet
    a = np.load(path+'data_'+str(index)+'.npy')
    a = a.reshape(2187,56,56,1).astype('float32') / 255.0
    b = np.load(path+'labels_'+str(index)+'.npy')
    x_train = a[0:train,:,:,:]
    x_test = a[train:,:,:,:]
    y_train = b[0:train,:]
    y_test = b[train:,:]
    return np.array(x_train), np.array(y_train), np.array(x_test),np.array(y_test)

x_train, y_train, x_test, y_test = next_load_data(path,1458, 165)

#arr = (340,469,656,859,1004,1246)
#긺, 넬, 덤, 랒, 묶, 뷕, 슛
for i in range(2,101):
#for i in arr:
    x_train0, y_train0, x_test0, y_test0 = next_load_data(path,1458, i)
    x_train = np.vstack((x_train,x_train0))
    y_train = np.vstack((y_train,y_train0))
    x_test = np.vstack((x_test,x_test0))
    y_test = np.vstack((y_test,y_test0))
    print('진행 : '+str(i))

#x_train = np.vstack((x_train,x_train))
#y_train = np.vstack((y_train,y_train))
#x_train = np.vstack((x_train,x_train))
#y_train = np.vstack((y_train,y_train))

with Back.tf.device('/gpu:0'):
    training = googlenet.fit(x_train, y_train,validation_data=(x_test,y_test),epochs=2, batch_size=56)
    now = datetime.now()
    temp = ('%s-%s-%s'%(now.year, now.month, now.day))
    googlenet.save('save/save_'+temp+'.h5')
    