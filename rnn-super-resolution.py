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

x           = Bi(GRU(500, dropout=0.1, recurrent_dropout=0.2, activation='relu', recurrent_activation='tanh', return_sequences=True))(input_tensor1)
#x           = Bi(GRU(300, dropout=0.1, recurrent_dropout=0.2, activation='relu', recurrent_activation='tanh', return_sequences=True))(x)
x           = TD(Dense(5000, activation='relu'))(x)
x           = BN()(x)
x           = TD(Dense(3000, activation='relu'))(x)
x           = BN()(x)
x           = TD(Dense(3000, activation='relu'))(x)
x           = BN()(x)
x           = TD(Dense(500, activation='relu'))(x)
x           = Dropout(0.10)(x)
decoded     = TD(Dense(1, activation='linear'))(x)

model       = Model(input_tensor1, decoded)
model.compile(optimizer=Adam(lr=0.0001, decay=0.03), loss='mae')

if '--train' in sys.argv:
  Xs, Ys = pickle.load(open('dataset.pkl', 'rb'))
  if '--resume' in sys.argv:
    model.load_weights(sorted(glob.glob('./models/000000049_0.000002000000.h5')).pop())
  print(Xs.shape)
  decay = 0.02
  init_rate =  0.00001
  for i in range(50):
    model.optimizer = Adam(lr=init_rate*(1.0 - decay*i))
    print("lr is {:.12f}".format(init_rate*(1.0 - decay*i)) )
    model.fit(Xs,Ys, shuffle=True, epochs=1, batch_size=300)
    model.save('models/{:09d}_{:.12f}.h5'.format(i,init_rate*(1.0 - decay*i)))

import itertools
from scipy.io import wavfile

if '--predict' in sys.argv:
  model.load_weights('models/000000019_00000.000002.h5')
  Xs, Ys = pickle.load(open('predict.pkl', 'rb'))

  if '--baseline' in sys.argv: 
    Xs = Xs*32766
    xs = np.array(list(itertools.chain(*Xs.tolist())), dtype=np.int16)   
    wavfile.write(f'xs_orig_{0}.wav', 4410*4, xs)
    
    Ys = Ys*32767
    ys = np.array(list(itertools.chain(*Ys.tolist())), dtype=np.int16)   
    wavfile.write(f'ys_orig_{0}.wav', 4410*4, ys)
  
  Yp = model.predict(Xs)
  
  Yp = Yp*32767
  yps = np.array(list(itertools.chain(*Yp.tolist())), dtype=np.int16)   
  wavfile.write(f'yp_orig_{0}.wav', 4410*4, yps)
   
