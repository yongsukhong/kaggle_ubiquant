
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#1.MNIST Data
from keras.datasets import mnist
(train_images, train_labels),(test_images,test_labels) = mnist.load_data()

print("train dataset shape:", train_images.shape, train_labels.shape)
print("test dataset shape:", test_images.shape, test_labels.shape)

print(train_images[0])

#2. MNIST Data Visualize

plt.imshow(train_images[0], cmap='gray')
plt.title(train_labels[0], size=20)
# plt.show()

ncols=5
figure,axs = plt.subplots(figsize=(10,5),nrows=1, ncols=ncols)

for i in range(ncols):
    axs[i].imshow(train_images[:5][i], cmap='gray')
# plt.show()

#3. MNIST Data Preprocessing
train_images = np.array(train_images/255.0, dtype=np.float32)
test_images = np.array(test_images/255.0, dtype =np.float32)

train_labels = np.array(train_labels, dtype=np.float32)
test_labels = np.array(test_labels, dtype =np.float32)

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

# label one hot encoding
from keras.utils.np_utils import to_categorical

train_labels_2 = to_categorical(train_labels)
test_labels_2 = to_categorical(test_labels)

# Dataset splitting

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(train_images, train_labels_2, test_size=0.3, random_state=1)

print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)

#4. ANN model structure
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Flatten(input_shape=([28,28])))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))

#5. Loss and Optimizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy

model.compile(optimizer = Adam(), loss ='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#6. ANN model fitting
epochs = 5
batch_size=32

history = model.fit(x=train_x, y=train_y, batch_size=batch_size, validation_data = (val_x, val_y), epochs=epochs, verbose=1)

print(history.history['loss'])
print(history.history['accuracy'])
print(history.history['val_loss'])
print(history.history['val_accuracy'])

#7. Model evaluate
model.evaluate(test_images, test_labels_2, verbose=0)

#8.  Comapre Model predicted value and real value
pred_y = model.predict(test_images).argmax(axis=1)
print(pred_y[:10])
print(test_labels[:10])

ncols=10
figure, axs = plt.subplots(figsize=(10,10), nrows=1, ncols=ncols)

for i in range(ncols):
    axs[i].imshow(test_images[:10][i], cmap='gray')

plt.show()

