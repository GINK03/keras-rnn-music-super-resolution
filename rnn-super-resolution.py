from keras.layers               import Input, Dense, GRU, LSTM, RepeatVector
from keras.models               import Model
from keras.layers.core          import Flatten
from keras.callbacks            import LambdaCallback 
from keras.optimizers           import SGD, RMSprop, Adam
from keras.layers.wrappers      import Bidirectional as Bi
from keras.layers.wrappers      import TimeDistributed as TD
from keras.layers               import merge
from keras.applications.vgg16   import VGG16 
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.merge         import Concatenate
from keras.layers.core          import Dropout
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re

input_tensor1 = Input(shape=(100, 1))

x           = Bi(LSTM(300, recurrent_dropout=0.05, recurrent_activation='tanh', return_sequences=True))(input_tensor1)


x           = Bi(LSTM(200, recurrent_dropout=0.05, return_sequences=True))(x)
x           = Dropout(0.10)(x)
x           = TD(Dense(2600, activation='relu'))(x)
x           = Dropout(0.10)(x)
x           = TD(Dense(2600, activation='relu'))(x)
x           = Dropout(0.10)(x)
decoded     = TD(Dense(1, activation='linear'))(x)

model       = Model(input_tensor1, decoded)
model.compile(optimizer=SGD(lr=0.005, decay=0.03, nesterov=False), loss='mae')

if '--train' in sys.argv:
  Xs, Ys = pickle.load(open('dataset.pkl', 'rb'))
  print(Xs.shape)
  model.fit(Xs,Ys, epochs=2, batch_size=500)
