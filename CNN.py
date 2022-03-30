# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:50:02 2021

@author: JoJo


# -*- coding: utf-8 -*-
"""

import keras_tuner as kt
import pickle 
# import keras 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential,optimizers,regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,BatchNormalization,GaussianNoise,Flatten,Dropout,Dense,Activation,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import csv
from collections import Counter
import time


with open('x_train.pkl', 'rb') as f:
    x_train_raw = pickle.load(f)

with open('x_test.pkl', 'rb') as f:
    x_test_raw = pickle.load(f)
    
with open('y_train.pkl', 'rb') as f:
    y_train_raw= pickle.load(f)

#Downscale

print(Counter(y_train_raw))

## Convert classes 
le=preprocessing.LabelEncoder()
le.fit(y_train_raw)
y_train=le.transform(y_train_raw)

# proprocessing 
scaler = StandardScaler()
# transform data
x_train_raw=np.asarray(x_train_raw)
x_train = scaler.fit_transform(x_train_raw.reshape(-1, x_train_raw.shape[-1])).reshape(x_train_raw.shape)

x_test_raw=np.asarray(x_test_raw)
x_test = scaler.fit_transform(x_test_raw.reshape(-1, x_test_raw.shape[-1])).reshape(x_test_raw.shape)

# x_train=np.divide(x_train_raw,255.0)
x_test=np.divide(x_test_raw,255.0)
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.20)

# shape 
N_train=np.shape(X_train)[0]
N_valid=np.shape(X_valid)[0]
N_test=np.shape(x_test)[0]
img_height=np.shape(X_train)[1]
img_width=np.shape(X_train)[2]
img_depth=1
input_shape_=(img_height,img_width,img_depth)

#reshape
X_train=X_train.reshape((N_train,img_height,img_width,img_depth))
X_valid=X_valid.reshape((N_valid,img_height,img_width,img_depth))
X_test=x_test.reshape((N_test,img_height,img_width,img_depth))

Y_train_OH=to_categorical(Y_train)
Y_valid_OH=to_categorical(Y_valid)
# %% 

def get_category(Y_OH):
    numerical_category=[np.argmax(y) for y in Y_OH]
    category=le.inverse_transform(numerical_category)
    return(category)

def visualize_data(images,category_OH):
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    for i in range(3 * 7):
        plt.subplot(3, 7, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        category=get_category(category_OH)
        plt.xlabel(category[i])
    plt.show()


# %% 
image_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=.2,
    height_shift_range=.2,
    horizontal_flip=False, 
shear_range=0, zoom_range=0.3, fill_mode="nearest")

# #training the image preprocessing
image_gen.fit(X_train)
batch_size=32
it = image_gen.flow(X_train, Y_train_OH, batch_size=batch_size)

batch_images, batch_labels = next(it)

print(Counter(get_category(batch_labels)))

visualize_data(batch_images,batch_labels)
np.shape(batch_images), print(batch_labels)

# Ajoute batch_size*iterations nombre d'exemples supplementaire au data
def augment_data(image_gen,batch_size,iterations,X_train,Y_train_OH):
    it = image_gen.flow(X_train, Y_train_OH,batch_size)
    for i in range(iterations):
        # Ajoute 200 examples des 11 classes avec une distribution aleatoire
        print(i)
        batch_images, batch_labels = next(it)
        X_train=np.concatenate((X_train,batch_images))
        Y_train_OH=np.concatenate((Y_train_OH,batch_labels))
    return X_train,Y_train_OH

aug_amount=0.3
aug_number=N_train*aug_amount
iterations=30
batch_size=int(aug_number/iterations)
X_train,Y_train_OH=augment_data(image_gen,batch_size,iterations,X_train,Y_train_OH)

image_gen = ImageDataGenerator(rotation_range=30,zoom_range=0.3,width_shift_range=0.3,height_shift_range=0.3,shear_range=0,horizontal_flip=True,
fill_mode="nearest")

#training the image preprocessing
image_gen.fit(X_train)
    
# %% Build the CNN Model 

def model_builder(hp):
    model_CNN2=Sequential()
    
    hp_units1 = hp.Choice('units1', values=[32,64])
    hp_units2 = hp.Choice('units2', values=[32,64,128])
    hp_units3 = hp.Choice('units3', values=[32,64,128])
    hp_units4 = hp.Choice('units4', values=[32,64,128,256])
    hp_units5 = hp.Choice('units5', values=[32,64,128,256])
    hp_units6 = hp.Choice('units6', values=[32,64,128,256,512])
    hp_units7 = hp.Float('noise', min_value=0, max_value=0.3, step=0.05)
    hp_units8 = hp.Float('drop_out', min_value=0, max_value=0.3, step=0.05)
    rate = hp.Choice('rate',values=[1e-2,1e-3,1e-4])
    beta1= hp.Choice('beta1',values=[0.8,0.9])
    
    drop_out=hp_units7
    noise=hp_units8
    
    model_CNN2.add(Conv2D(hp_units1,(3,3),padding='same',strides=(1,1),input_shape=(96,96,1)))
    model_CNN2.add(BatchNormalization())
    model_CNN2.add(Activation("relu"))
    model_CNN2.add(MaxPooling2D((2,2)))
    model_CNN2.add(GaussianNoise(noise))
    model_CNN2.add(Dropout(drop_out))
    
    model_CNN2.add(Conv2D(hp_units2,(3,3)))
    model_CNN2.add(BatchNormalization())
    model_CNN2.add(Activation("relu"))
    model_CNN2.add(MaxPooling2D((2,2)))
    model_CNN2.add(GaussianNoise(noise))
    model_CNN2.add(Dropout(drop_out))
    
    model_CNN2.add(Conv2D(hp_units3,(3,3)))
    model_CNN2.add(BatchNormalization())
    model_CNN2.add(Activation("relu"))
    model_CNN2.add(MaxPooling2D((2,2)))
    model_CNN2.add(GaussianNoise(noise))
    model_CNN2.add(Dropout(drop_out))
    
    model_CNN2.add(Conv2D(hp_units4,(3,3)))
    model_CNN2.add(BatchNormalization())
    model_CNN2.add(Activation("relu"))
    model_CNN2.add(MaxPooling2D((2,2)))
    model_CNN2.add(GaussianNoise(noise))
    model_CNN2.add(Dropout(drop_out))
    
    model_CNN2.add(Conv2D(hp_units5,(3,3),kernel_regularizer = regularizers.l2( l=0.01)))
    model_CNN2.add(BatchNormalization())
    model_CNN2.add(Activation("relu"))
    model_CNN2.add(MaxPooling2D((2,2)))
    model_CNN2.add(GaussianNoise(noise))
    model_CNN2.add(Dropout(drop_out))
    
    model_CNN2.add(Flatten())
    model_CNN2.add(Dense(hp_units6,Activation("relu")))
    model_CNN2.add(Dropout(drop_out))
    model_CNN2.add(Dense(11, Activation("softmax")))
    model_CNN2.summary()
    
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model_CNN2.compile(loss="categorical_crossentropy", 
                 optimizer=optimizers.Adam(learning_rate=rate,beta_1=beta1),
                  metrics=["accuracy"])

    return model_CNN2

t0=time.time()
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='new',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
epochs=150
tuner.search(image_gen.flow(X_train, Y_train_OH, batch_size=128), validation_data=(X_valid, Y_valid_OH),
         steps_per_epoch=len(X_train)/128, epochs=epochs, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps2=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
      Without augemnted data
best units1 is  {best_hps2.get('units1')}
best units2 is  {best_hps2.get('units2')}
best units3  is  {best_hps2.get('units3')}
best units4  is  {best_hps2.get('units4')}
best units5  is  {best_hps2.get('units5')}
best units6  is  {best_hps2.get('units6')}
best noise  is  {best_hps2.get('noise')}
best drop_out is  {best_hps2.get('drop_out')}
best rate is  {best_hps2.get('rate')}
best beta1 is  {best_hps2.get('beta1')}
""")

t=time.time()-t0
# build the model with best parameters
model_CNN2=model_builder(best_hps2)

# %% Learn

checkpoint = ModelCheckpoint(filepath="models/my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5", 
                             monitor="val_loss",
                             verbose=1, 
                             save_best_only=True,
                             mode="min")
epochs=50
callbacks = [checkpoint]
history=model_CNN2.fit(image_gen.flow(X_train, Y_train_OH, batch_size=128), validation_data=(X_valid, Y_valid_OH),
         steps_per_epoch=len(X_train)/128, epochs=epochs, shuffle=True,callbacks=callbacks)

#Plot learning
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.plot(N, history.history["accuracy"], label="train_acc")
plt.plot(N, history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

model_CNN2.load_weights("models/my_best_model.epoch822-loss0.88.hdf5")
y_pred_v=np.argmax(model_CNN2.predict(X_valid),axis=1)
print(f1_score(Y_valid,y_pred_v,average="macro"))


# %% Make predictions and sumbit


y_pred=model_CNN2.predict(X_test)
y_pred=np.asarray([np.argmax(y) for y in y_pred])

def submit_data(y_pred):
    with open('submission_new69.csv', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["ID", "class"])
        for i in range(y_pred.shape[0]):
            writer.writerow([i,y_pred[i]])
            
submit_data(y_pred)

# %% Confusion Matrix

from mpl_toolkits.axes_grid1 import make_axes_locatable
 
def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()
 
    np.fill_diagonal(cmx_zero_diag, 0)
 
    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(12, 3)
    [a.set_xticks(range(len(cmx)+1)) for a in ax]
    [a.set_yticks(range(len(cmx)+1)) for a in ax]
         
    im1 = ax[0].imshow(cmx, vmax=vmax1)
    ax[0].set_title('as is')
    im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
    ax[1].set_title('%')
    im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
    ax[2].set_title('% and 0 diagonal')
 
    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.tight_layout()
    

y_valid_pred=model_CNN2.predict(X_valid)
y_valid_pred=np.asarray([np.argmax(y) for y in y_valid_pred])

cmx=confusion_matrix(Y_valid, y_valid_pred)    
plot_confusion_matrix(cmx)
 
# the types appear in this order
categs=sorted(np.unique(get_category(Y_valid_OH)));
print('\n', sorted(np.unique(get_category(Y_valid_OH))))

