from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Activation
from keras.layers import MaxPooling2D,UpSampling2D
from keras.layers import Dropout,Dense,Flatten,BatchNormalization
from keras.optimizers import *
from keras.models import load_model
from keras import regularizers
import os
import cv2
import numpy as np

angles = range(-2,3)
shifts = [[0,0],[0,1],[1,0],[1,1],[0,2],[2,0],[1,2],[2,1],[2,2],
                [0,-1],[-1,0],[-1,-1],[0,-2],[-2,0],[-1,-2],[-2,-1],[-2,-2],
                [1,-1],[1,-2],[2,-1],[2,-2],
                [-1,1],[-1,2],[-2,1],[-2,2]]
multiplier = len(angles)*len(shifts)
X_train=np.zeros((100*multiplier,128,128,3))
y_train=np.zeros((100*multiplier,128,128,3))

path_x = 'Data/X/'
path_y = 'Data/Y2/'
total = 0
#for pos in range(len(path_x)):
for img in os.listdir(path_x):
    originalIm = cv2.imread(path_x+img)
    segmentedIm = cv2.imread(path_y+img)

    for angle in angles:
        for shift in shifts :

            M = cv2.getRotationMatrix2D((128/2,128/2),angle,1)
            shiftM = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
            rotatedIm = cv2.warpAffine(originalIm,M,(128,128))
            rotatedSegmentedIm = cv2.warpAffine(segmentedIm,M,(128,128))
            rotatedShiftedIm = cv2.warpAffine(rotatedIm,shiftM,(128,128))
            rotatedSegmentedShiftedIm = cv2.warpAffine(rotatedSegmentedIm,shiftM,(128,128))
            X_train[total]=rotatedShiftedIm
            y_train[total]=rotatedSegmentedShiftedIm
            total+=1

X_test = np.zeros((5,128,128,3))
tests = ["A-train0101.jpg","A-train0102.jpg","A-train0103.jpg","A-train0104.jpg","A-train0105.jpg"]
for pos in range(len(tests)):
    X_test[pos] = cv2.imread(path_x+tests[pos])

#
# meen = np.mean(X_train,axis=(0,1,2))
# std = np.std(X_train,axis=(0,1,2))
# X_train-=meen
# X_train/=std
#
# #y_train-=meen
# y_train/=255
#

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
clf = Sequential()

clf.add(Convolution2D(filters=64,kernel_size=(3,3),input_shape=(128,128,3), padding='same'))
clf.add(BatchNormalization())
clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))

clf.add(Convolution2D(filters=128,kernel_size=(3,3),padding='same'))
clf.add(BatchNormalization())
clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))

clf.add(Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
clf.add(BatchNormalization())
clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))

clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
clf.add(BatchNormalization())
clf.add(Activation('relu'))

clf.add(Convolution2D(filters=512,kernel_size=(3,3), padding='same'))
clf.add(BatchNormalization())
clf.add(Activation('relu'))

clf.add(UpSampling2D((2,2)))
clf.add(Convolution2D(filters=256,kernel_size=(3,3), padding='same'))
clf.add(BatchNormalization())
clf.add(Activation('relu'))

clf.add(UpSampling2D((2,2)))
clf.add(Convolution2D(filters=128,kernel_size=(3,3), padding='same'))
clf.add(BatchNormalization())
clf.add(Activation('relu'))

clf.add(UpSampling2D((2,2)))
clf.add(Convolution2D(filters=64,kernel_size=(3,3), padding='same'))
clf.add(BatchNormalization())
clf.add(Activation('relu'))

clf.add(Convolution2D(3, (3, 3), padding='same'))
clf.add(Activation('softmax'))

clf.compile(optimizer=adam,loss='mse',metrics=['mae'])
clf.fit(X_train,y_train,batch_size=20, epochs=1000,validation_split=0.1)

y_out = clf.predict(X_test)
for y in range(y_out.shape[0]):
    cv2.imwrite('y'+str(y)+'.jpg',y_out[y])
