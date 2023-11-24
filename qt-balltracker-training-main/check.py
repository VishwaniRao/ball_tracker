from __future__ import print_function

import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from numpy.random import choice
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
from keras.models import Sequential
import tensorflow as tf
import keras.backend as K
import random
import keras

## x,y,r

x = []
y = []
for i in range(100000):
    time_steps = random.randint(1,10)
    time_steps = 3

    features = []
    for j in range(time_steps):
        features.append(array([random.randint(-100,100),random.randint(-100,100),random.randint(-100,100)]))

    out_ = []
    for single in features:
        out_.append(array(np.sum(single)))

    x.append(features)
    y.append(out_)

    
    # features = array(features)
    # out_ = array(out_)
    # y.append(out_.reshape(1,time_steps,1))
    # x.append(features.reshape(1,time_steps,3))

x = array(x)
y=array(y)

total_samples = len(x)
test_number = int(0.2*total_samples)
train_number = int(0.8*total_samples)

# train_x = tf.ragged.constant(x[:train_number])
# train_y = tf.ragged.constant(y[:train_number])
train_x = tf.convert_to_tensor(x[:train_number])
train_y = tf.convert_to_tensor(y[:train_number])
test_x = x[train_number:]
test_y = y[train_number:]

print(train_x.shape)
print(train_y.shape)
print(train_x[0])
print(train_y[0])
# a = tf.ragged.constant(train_x)
# print(a.shape)
# print(len(train_x),len(train_y))
# print(len(test_x),len(test_y))


I = Input(shape=(3, 3)) # unknown timespan, fixed feature size
# lstm = LSTM(1,return_sequences=True,dropout=0.1,activation='tanh')
lstm = LSTM(3,dropout=0.1,activation='tanh')
Model = Sequential()
Model.add(I)
Model.add(lstm)
Model.add(Dense(3))

# # model = Model(inputs=[I], outputs=[lstm(I)])
optimizer = keras.optimizers.Adam(lr=0.00001)
scce = tf.keras.losses.SparseCategoricalCrossentropy()
sample_input = array([[10,10,5],[10,10,5],[10,10,5]]).reshape(1,3,3)
sample_output = array([25,25,25]).reshape(1,1,3)
Model.compile(loss="mse", optimizer=optimizer)
Model.fit(train_x,train_y,128, 100000)
# Model.fit(sample_input,sample_output, 1, 1000)
# print(Model.predict(sample_input))
# print(train_x[0].shape)
# print(train_y[0])
# print()
print(Model.predict(sample_input))


# new_train_x = train_x[:2]
# new_train_y = train_y[:2]
# Model.fit(new_train_x,new_train_y,1,10)
# for i in range(10):
#     print(model.predict(train_x[i]).shape)
#     print(train_y[i].shape)

# print("yyyyyyy")
# print(train_y[0])