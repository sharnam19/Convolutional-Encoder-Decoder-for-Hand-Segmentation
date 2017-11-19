from keras.models import load_model
import os
import cv2
import numpy as np
import sys

#clf.save('model-10.h5')
clf = load_model('model-10.h5')
test_folder = "New Test/"
test_save_folder = "New Test Output/"
length = len(os.listdir(test_folder))
X = np.zeros((length,128,128,3))
read=[]
total=0
for img in os.listdir(test_folder):
    X[total]=cv2.imread(test_folder+img)
    total+=1
    read.append(img)

X-=128.0
X/=128.0
y_out = clf.predict(X)
y_out*=128.0
y_out+=128.0
for y in range(len(read)):
    cv2.imwrite(test_save_folder+read[y],y_out[y])
