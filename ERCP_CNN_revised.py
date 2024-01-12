c# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:05:46 2020

@author: NTCU_KLAB
"""

# Importing the necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout , BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import EarlyStopping
import cv2
import os

# Description of the Pneumonia Dataset
labels = ['ERCP', 'NORMAL']
img_size = 128
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

# Loading the Dataset    
train = get_training_data('../DeepLearning/ERCP_Data/Refined_dataset/train')
test = get_training_data('../DeepLearning/ERCP_Data/Refined_dataset/test')

# Creating a target list which will contain target labels
# Data Visualization & Preprocessing

l = []
for i in train:
    if (i[1] == 0):
        l.append("ERCP")
    else:
        l.append("NORMAL")

sns.set_style('darkgrid')
sns.countplot(l) 

plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])

x_train = []
y_train = []

x_test = []
y_test = []

for feature, label in train: 
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
    
    
# Normalize the data
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255

# resize data for deep learning 
# resize data for deep learning 
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)


x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

# define generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

# fit generator on our train features
datagen.fit(x_train)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

# 建立模型
# STEP1. 建立卷積層與池化層    
model = Sequential()  
# Create CN layer 1  
model.add(Conv2D(filters=64,  
                 kernel_size=(3,3),  
                 padding='same',  
                 input_shape=(img_size , img_size, 1),  
                 activation='relu'))  
# Create Max-Pool 1  
model.add(MaxPooling2D(pool_size=(2,2)))  
# Add Dropout layer  
model.add(Dropout(0.2))  


# Create CN layer 2  
model.add(Conv2D(filters=64,  
                 kernel_size=(3,3),  
                 padding='same',
                 activation='relu'))    
# Create Max-Pool 2  
model.add(MaxPooling2D(pool_size=(2,2)))    
# Add Dropout layer  
model.add(Dropout(0.3))

# Create CN layer 3  
model.add(Conv2D(filters=128,  
                 kernel_size=(3,3),  
                 padding='same',
                 activation='relu'))    
# Create Max-Pool 3  
model.add(MaxPooling2D(pool_size=(2,2)))    
# Add Dropout layer  
model.add(Dropout(0.3))

# Create CN layer 3  
model.add(Conv2D(filters=128,  
                 kernel_size=(3,3),  
                 padding='same',
                 activation='relu'))    
# Create Max-Pool 3  
model.add(MaxPooling2D(pool_size=(2,2)))    
# Add Dropout layer  
model.add(Dropout(0.3))

# Create CN layer 4  
model.add(Conv2D(filters=256,  
                 kernel_size=(3,3),  
                 padding='same',
                 activation='relu'))    
# Create Max-Pool 4  
model.add(MaxPooling2D(pool_size=(2,2)))    
# Add Dropout layer  
model.add(Dropout(0.3))

# STEP2. 建立神經網路 
# - 建立平坦層
model.add(Flatten())


# - 建立 Hidden layer 
model.add(Dense(512, activation='relu'))  
model.add(Dropout(0.5))
# - 建立輸出層
model.add(Dense(1, activation='sigmoid')) 

# STEP3. 查看模型的摘要 
model.summary()  
print("") 


optimizer = Adam(lr=0.0001, decay=1e-5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
epoch=200
callback = EarlyStopping(monitor='loss', patience=6)
history = model.fit(datagen.flow(x_train,y_train, batch_size=32), validation_data=(x_val, y_val), epochs = epoch, verbose = 2, callbacks=[callback])

print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

# Analysis after Model Training
epochs = [i for i in range(200)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

predictions = model.predict_classes(x_test)
y_pred_prob = model.predict(x_test)
predictions = predictions.reshape(1,-1)[0]
predictions[:15]

cm = confusion_matrix(y_test,predictions)
sns.set(font_scale=1.5)
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False, linewidth = 1, xticklabels = labels, yticklabels = labels)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('true label')
plt.xlabel('predicted label');
plt.draw()
print(classification_report(y_test, predictions, target_names = ['ERCP (Class 0)','Normal (Class 1)']))