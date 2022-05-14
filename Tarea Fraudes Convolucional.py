import csv
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile,isdir, join
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv1D, Dropout,Activation,MaxPooling1D,Flatten
from tensorflow.keras.optimizers import RMSprop
from scipy.fftpack import fft
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("creditcard.csv")

caracteristicas = df.drop("Class",axis=1)
etiquetas = df.loc[:,['Class']]

#Hacemos Oversampling para balancear nuestros datos y la red pueda entrenar de una maner muchisimo más eficiente
x_data=df.drop(['Class'], axis=1)
y_data=df['Class']
print(x_data.shape)
print(y_data.shape)
ros = RandomOverSampler(sampling_strategy=1)
x_res, y_res = ros.fit_resample(x_data,y_data)

print(x_res.shape)
print(y_res.shape)


x_train,x_test,y_train,y_test = train_test_split(x_res,y_res,test_size=0.3,random_state=1)
print(y_train.shape)
print(x_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')




model = Sequential()
model.add(Conv1D(5, 6, input_shape=(30,1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(25))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=400,
                    epochs=30,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print(score)

loss=history.history['loss']
val_loss= history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'y', label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

acc=history.history['accuracy']
val_acc= history.history['val_accuracy']
plt.plot(epochs,acc,'y', label='Training acc')
plt.plot(epochs,val_acc,'r',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

