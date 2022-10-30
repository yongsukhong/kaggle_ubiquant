import pandas as pd
import numpy as np


#1. MPG Dataset
dataset = pd.read_csv('mpg.csv')
dataset.head()

#2. Data preprocessing
dataset.info()

dataset.dropna(inplace=True)
dataset.info()

x = dataset.iloc[:,0:-1] #feature
y = dataset.iloc[:-1] #label

# Data scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
xs = scaler.fit_transform(x)
x = pd.DataFrame(xs, columns=x.columns)

# Data seperating
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2)

#3. ANN modeling

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation='relt'))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])

model.summary()

#4. Model training
epochs = 10
batch_size = 16

history = model.fit(train_x, train_y, epochs=epochs, batch_size = batch_size, validation_split = 0.3, verbose=1)

#5. Data Evaluating
from sklearn.metrics import r2_score,mean_squared_error

y_pred = model.predict(test_x)
R2 = r2_score(test_y, y_pred)
RMSE = mean_squared_error(test_y, y_pred)**(1/2)

print('test data evaluation', R2, RMSE)

print(test_y[:10], y_pred[:10])