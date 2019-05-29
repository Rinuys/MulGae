# https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff
from keras import layers
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from keras.initializers import glorot_uniform
from datetime import datetime
import keras.backend.tensorflow_backend as Back
import numpy as np
import matplotlib.pyplot as plt

def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X
    
    X = layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', 
        name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', 
        name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', 
        name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid', 
        name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', 
        name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', 
        name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    
    X_shortcut = layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), 
        padding='valid', name=conv_name_base + '1', 
        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3, 
        name=bn_name_base + '1')(X_shortcut)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X
    
img_shape = (56, 56, 1)
def ResNet50(input_shape = img_shape, classes = 2350):
    
    # Define the input as a tensor with shape input_shape
    X_input = layers.Input(input_shape)
    
    # Zero-Padding
    X = layers.ZeroPadding2D((2, 2))(X_input)
    #X = X_input
    
    # Stage 1
    X = layers.Conv2D(64, (7,7), strides = (2,2), name = 'conv1', 
        kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3, 3), strides=(2,2), padding='same')(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [56, 56, 112], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [56, 56, 112], stage=2, block='b')
    X = identity_block(X, 3, [56, 56, 112], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[112, 112, 448], stage=3, block='a', s=2)
    X = identity_block(X, 3, [112, 112, 448], stage=3, block='b')
    X = identity_block(X, 3, [112, 112, 448], stage=3, block='c')
    X = identity_block(X, 3, [112, 112, 448], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[224, 224, 896], stage=4, block='a', s=2)
    X = identity_block(X, 3, [224, 224, 896], stage=4, block='b')
    X = identity_block(X, 3, [224, 224, 896], stage=4, block='c')
    X = identity_block(X, 3, [224, 224, 896], stage=4, block='d')
    X = identity_block(X, 3, [224, 224, 896], stage=4, block='e')
    X = identity_block(X, 3, [224, 224, 896], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[448, 448, 1792], stage=5, block='a', s=2)
    X = identity_block(X, 3, [448, 448, 1792], stage=5, block='b')
    X = identity_block(X, 3, [448, 448, 1792], stage=5, block='c')

    # AVGPOOL
    X = layers.AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = layers.Flatten()(X)
    X = layers.Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.Reshape((2350,), input_shape=(1,1,2350))(X)
    X = layers.Activation('softmax')(X)
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

resnet = ResNet50(img_shape)
resnet.summary()

print("전체 파라미터 수 : {}".format(sum([arr.flatten().shape[0] for arr in resnet.get_weights()])))

with Back.tf.device('/gpu:0'):
    # googlenet.compile(optimizer=SGD(lr=0.01,decay=1e-6, momentum=0.9), 
    # loss='categorical_crossentropy', metrics=["accuracy"])
    resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

path = 'D:\PHD08\phd08-conversion-master\phd08_npy_results\phd08_'
def next_load_data(path, train, index):
    #DataSet
    try:
        a = np.load(path+'data_'+str(index)+'.npy')
    except:
        return -1
    a = a.reshape(2187,56,56,1).astype('float32') / 255.0
    b = np.load(path+'labels_'+str(index)+'.npy')
    x_train = a[0:train,:,:,:]
    x_test = a[train:,:,:,:]
    y_train = b[0:train,:]
    y_test = b[train:,:]
    return np.array(x_train), np.array(y_train), np.array(x_test),np.array(y_test)

#resnet.load_weights('save/save_resnet_2019-5-29.h5')

######################## train ###########################

with Back.tf.device('/gpu:0'):
    for i in range(1):
        state = open("state.txt","r")
        r = int(state.readline())
        x_train, y_train, x_test, y_test = next_load_data(path,1458, r)
        print('Data_Loading...'+str(r))
        r+=1

        for i in range(r,99+r):
            try:
                x_train0, y_train0, x_test0, y_test0 = next_load_data(path,1458, i)
            except:
                break
            
            x_train = np.vstack((x_train,x_train0))
            y_train = np.vstack((y_train,y_train0))
            x_test = np.vstack((x_test,x_test0))
            y_test = np.vstack((y_test,y_test0))
            print('Data_Loading...'+str(i))

        state.close()

        state = open("state.txt","w+")

        training = resnet.fit(x_train, y_train, epochs=4, batch_size=56, shuffle=True)
        now = datetime.now()
        temp = ('%s-%s-%s'%(now.year, now.month, now.day))
        resnet.save_weights('save/save_resnet_'+temp+'.h5')

        state.write(str(r+99))
        state.close()

######################## test ###########################

# r = 100
# t = 20

# x_train, y_train, x_test, y_test = next_load_data(path,1458, r)
# print('Data_Loading...'+str(r))
# r+=1

# for i in range(r,t+r):
#     try:
#         x_train0, y_train0, x_test0, y_test0 = next_load_data(path,1458, i)
#     except:
#         break
#     x_train = np.vstack((x_train,x_train0))
#     y_train = np.vstack((y_train,y_train0))
#     x_test = np.vstack((x_test,x_test0))
#     y_test = np.vstack((y_test,y_test0))
#     print('Data_Loading...'+str(i))

# with Back.tf.device('/gpu:0'):
#     evaluate = googlenet.evaluate(x_test, y_test, batch_size=56)
#     print(evaluate)

######################## predict ###########################

# import cv2
# x = cv2.imread('test_images/1.png', cv2.IMREAD_GRAYSCALE)
# x = x/255
# for i in range(len(x)):
#     for j in range(len(x[0])):
#         if (x[i][j]==0):
#             x[i][j] = 1
#         else:
#             x[i][j] = 0

# #cv2.imshow('gray',x)
# x = x.reshape(1,56,56,1)


# with Back.tf.device('/gpu:0'):
#     x_predict = resnet.predict(x, batch_size=1)
#     print(x_predict.shape)

# x_predict = x_predict[0]

# m = 0
# for i in range(len(x_predict)):
#     if(x_predict[i]>x_predict[m]): 
#         m = i
# print(m)
# x_train, y_train, x_test, y_test = next_load_data(path, 1458, m+1)

# result_image = x_test[0].reshape(56,56)
# cv2.imshow('gray',result_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()