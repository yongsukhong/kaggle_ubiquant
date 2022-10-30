#import library

import numpy as np
import pandas as pd

#1. Dataset

from tensorflow.keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)
#image size 32 *32, RGB 3dimension

#2. EDA
print(train_images[0,:,:,:], train_labels[0,:])

NAMES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']) #label name
print(train_labels[:10])

import matplotlib.pyplot as plt
ncols=8
figure, axs = plt.subplots(figsize=(22,6), nrows=1, ncols=ncols)

for i in range(ncols):
    axs[i].imshow(train_images[:8][i], cmap='gray')

# Data preprocessing
train_images = np.array(train_images/255.0, dtype = np.float32)
test_images = np.array(test_images/255.0, dtype = np.float32)

train_labels = np.array(train_labels, dtype = np.float32)
test_labels = np.array(test_labels, dtype = np.float32)

#print(train_images[0,:,:,:])

train_labels = train_labels.squeeze()
test_labels = test.labels.squeeze()

train_labels

#4. CNN modeling 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D , GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam , RMSprop 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint , LearningRateScheduler

input_tensor = Input(shape=(32, 32, 3))

x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2)(x)

x = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(x)
x = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=2)(x)

x = Flatten(name='flatten')(x)
x = Dropout(rate=0.5)(x)
x = Dense(300, activation='relu', name='fc1')(x)
x = Dropout(rate=0.3)(x)
output = Dense(10, activation='softmax', name='output')(x)

model = Model(inputs=input_tensor, outputs=output)

# model.summary()

# Since we label-encoded instead of one-hot-encoding. we must designate sparse_categorical_crossentropy for loss function
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=train_images, y=train_labels, batch_size=64, epochs=30, validation_split = 0.15)

#5. model evaluation

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.yticks(np.arange(0,1,0.05))
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'], label='valid')
plt.legend()

model.evaluate(test_images, test_labels)

#6. Predicting one data
preds = model.predict(np.expand_dims(test_images[0], axis=0))
print('예측 결과 shape:', preds.shape)
print('예측 결과:', preds.argmax())

plt.imshow(test_images[0])
pred_y = model.predict(test_images).argmax(axis=1)
print(pred_y[:5])
print(test_labels[:5])

#
ncols = 5
figure, axs =plt.subplots(figsize=(20,20), nrows=1, ncols=ncols)

for i in range(ncols):
    axs[i].imshow(test_images[:5][i], cmap='gray')

#Pre-trained model
from tensorflow.keras.applications import VGG16, ResNet50, ResNet50V2, Xception

model = VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
model.summary()

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D , GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam , RMSprop 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint , LearningRateScheduler

base_model = VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
bm_output = base_model.output

# base model의 output을 입력으로 CIFAR10용 Classification layer를 재 구성. 
x = GlobalAveragePooling2D()(bm_output)
x = Dropout(rate=0.5)(x)
x = Dense(50, activation='relu', name='fc1')(x)
x = Dropout(rate=0.2)(x)
output = Dense(10, activation='softmax', name='output')(x)

model = Model(inputs=base_model.input, outputs=output)

# model.summary()
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_images, y=train_labels, batch_size=64, epochs=1, validation_split=0.15 )
