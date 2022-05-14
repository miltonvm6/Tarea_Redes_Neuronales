import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

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

num_classes=2
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Dense(60, activation='relu', input_shape=(30,)))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

file_name = 'My_saved_model'

tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))
history = model.fit(x_train, y_trainc,
                    batch_size=128,
                    epochs=30,
                    verbose=1,
                    validation_data=(x_test, y_testc),
                    callbacks=[tensorboard])

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


score = model.evaluate(x_test, y_testc, verbose=1)
a=model.predict(x_test)
#b=model.predict_proba(x_test)
print(a.shape)
print(a[1])
#Para guardar el modelo en disco
model.save("red.h5")
#para cargar la red:
modelo_cargado = tf.keras.models.load_model('red.h5')