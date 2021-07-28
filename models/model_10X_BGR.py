# -*- coding: utf-8 -*-
"""
Created on Thu May 27 00:03:44 2021

@author: fbasatemur
"""

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.models import load_model
from keras import layers
from matplotlib import pyplot as plt
import numpy as np
import cv2 
import os
from keras.callbacks import TensorBoard
import time

time = time.strftime("%Y_%m_%d_%H_%M_%S")
kerasboard = TensorBoard(log_dir="./tensorboard_logs/{}".format(time),
                        histogram_freq=1,
                        batch_size=32,
                        write_grads=False)

#%% ANN model
def CreateModel(w,h,c):
      
      model = Sequential()

      model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='valid', input_shape=(w,h,c)))
      model.add(layers.MaxPooling2D((2, 2)))
      #model.add(BatchNormalization())
      
      model.add(layers.Conv2D(16, (3, 3), activation='relu'))
      #model.add(Dropout(0.5))
      #model.add(Activation('relu'))
      model.add(layers.MaxPooling2D((2, 2)))     
      model.add(layers.Flatten())                             

      model.add(Dense(16, activation="relu"))
      model.add(BatchNormalization())
      model.add(Dense(1, activation="sigmoid"))             # unipolar sigmoid (0-1)
      #model.compile(optimizer='rmsprop', loss='binary_crossentropy')
      model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
      #model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
      
      return model

def ShowAccuracy(history):
      
      for key in history.history.keys():  
            plt.plot(history.history[key], label = key)
    
      plt.xlabel("epochs")
      plt.ylabel("values")
      
      plt.legend()
      plt.show()
      

def Process(img):
      #bgr
      arry = img

      h = img.shape[0]
      w = img.shape[1]

      arry = np.delete(arry, range(1,w,2), 0)
      arry = np.delete(arry, range(1,h,2), 1)

      return arry
      

def Normalize(img):
      return np.array(img/255.0, dtype=np.float32)

import re
import glob

def numericalSort(value):
      numbers = re.compile(r'(\d+)')
      parts = numbers.split(value)
      parts[1::2] = map(int, parts[1::2])
      return parts

def ReadImages(folder_path, images, image_names):
      for filename in sorted(glob.glob(folder_path + '*.bmp'), key=numericalSort):
            image_names.append(filename.split('\\')[-1])
            img = cv2.imread(filename)
            if img is not None:
              images.append(Normalize(Process(img)))
              if(len(images) % 100 == 0):
                    print(len(images))

def ReadLabels(label_path, labels):
      with open(label_path, "r") as fileLabel:
          for line in fileLabel:
              #labels.append(int(line.split()[0]))
              labels.append(float(line.split()[0]))
      
      fileLabel.close() 
      
#%%
# load the images
folder_path1 = ".\\DATASET - BMP\\237-15PAP_10x_1024_1024\\"
folder_path3 = ".\\DATASET - BMP\\273-17P_10x_1024_1024\\"
folder_path5 = ".\\DATASET - BMP\\546-14 Giemse_10x_1024_1024\\"

images = []
image_names = []

# 10X
ReadImages(folder_path1, images, image_names)
ReadImages(folder_path3, images, image_names)
ReadImages(folder_path5, images, image_names)


print("all images imported")

#%%
label_path1 = ".\\LABELS\\237-15PAP_10x_labels.txt"
label_path3 = ".\\LABELS\\273-17P_10x_labels.txt"
label_path5 = ".\\LABELS\\546-14 Giemse_10x_labels.txt"

labels = []

# 10X
ReadLabels(label_path1, labels)
ReadLabels(label_path3, labels) 
ReadLabels(label_path5, labels)


#%%
# labels numerated with images index
numeratedLabels = np.vstack((labels, np.arange(len(labels)))).T

#%%
from sklearn.model_selection import train_test_split

images = np.array(images)
x_train, x_test, y_train, y_test = train_test_split(images, numeratedLabels, test_size=0.2, random_state=0)

print("Total images: " + str(numeratedLabels.shape[0]))
print("Total train images: " + str(y_train.T[0].shape[0]))
print("Total test images: " + str(y_test.T[0].shape[0]))

#%%
h = images.shape[1]
w = images.shape[2]
c = images.shape[3]
model = CreateModel(w, h, c)
print(model.summary())

#%%

history = model.fit(x_train, y_train.T[0], epochs=50, batch_size=32, callbacks=[kerasboard])

print("tensorboard --logdir="+kerasboard.log_dir)

#%%
ShowAccuracy(history)
print(model.evaluate(x_test, y_test.T[0]))

#%% y_test_predict print

y_test_predict = model.predict(x_test)
#%%
TH = 0.5
print("  ----Predict----    -----Real-----  ---Images Tags - Paths---  --ERROR > TH--")
for i in range(len(y_test_predict)):
     print( str(i) + " : "+ str(y_test_predict[i]) + "\t\t\t" + str(y_test.T[0][i]) + "\t\t\t\t" + str(int(y_test.T[1][i])) + " - " + image_names[int(y_test.T[1][i])], end='')
     print("\t\t\t X" if abs(y_test.T[0][i] - y_test_predict[i])[0] > TH  else "")


#%% model save .json

model_json = model.to_json()

with open("./model_save_10X_BGR.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("./model_save_10X_BGR.h5")


#%% model load

json_file = open('./model_save_10X_BGR.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("./model_save_10X_BGR.h5")


#%% loaded_model predict

loaded_model_predict = loaded_model.predict(x_test)

print(" ----Predict----      -------Real-------    -----Images Tags - Paths----- ")
for i in range(len(loaded_model_predict)):
     print( str(i) + " : "+ str(loaded_model_predict[i]) + " \t\t     " + str(y_test.T[0][i]) + " \t\t\t\t       " + str(int(y_test.T[1][i])) + " - " + image_names[int(y_test.T[1][i])])


#%% each layer outputs
from keras import backend as K

outputs = [K.function([loaded_model.input], [layer.output])([x_test, 1]) for layer in loaded_model.layers]

#%% print min and max values

print("Min value:" + str(np.amin(loaded_model_predict)) + " index: " + str(np.argmin(loaded_model_predict, axis=0)))
print("Max value:" + str(np.amax(loaded_model_predict)) + " index: " + str(np.argmax(loaded_model_predict, axis=0)))