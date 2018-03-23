from keras.layers               import Input, Dense, GRU, LSTM, CuDNNLSTM, RepeatVector
from keras.models               import Model
from keras.layers.core          import Flatten
from keras.callbacks            import LambdaCallback 
from keras.optimizers           import SGD, RMSprop, Adam
from keras.layers.wrappers      import Bidirectional as Bi
from keras.layers.wrappers      import TimeDistributed as TD
from keras.layers               import merge
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.merge         import Concatenate
from keras.layers.core          import Dropout
from keras.layers.merge         import Concatenate as Concat
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.merge         import Dot,Multiply
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re
import time

input_tensor1 = Input(shape=(50, 16))
x           = Bi(CuDNNLSTM(300, return_sequences=True))(input_tensor1)
x           = TD(Dense(500, activation='relu'))(x)
x           = Bi(CuDNNLSTM(300, return_sequences=True))(x)
x           = TD(Dense(500, activation='relu'))(x)
x           = TD(Dense(20, activation='relu'))(x)
decoded     = Dense(1, activation='linear')(x)
print(decoded.shape)
model       = Model(input_tensor1, decoded)
model.compile(RMSprop(lr=0.0001, decay=0.03), loss='mae')

buff = None
now  = time.strftime("%H_%M_%S")
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
batch_callback = LambdaCallback(on_epoch_end=lambda batch,logs: callback(batch,logs) )


if '--train' in sys.argv:
  Xs, Ys = np.load('Xs.npy'), np.load('Ys.npy')
  split = int(len(Xs)*0.8)
  Xs, Xst, Ys, Yst = Xs[:split], Xs[split:], Ys[:split], Ys[split:]
  if '--resume' in sys.argv:
    model.load_weights(sorted(glob.glob('./models/*.h5')).pop(0))
  print(Xs.shape)
  decay = 0.01
  init_rate =  0.0003
  for i in range(100):
    lr = init_rate*(1.0 - decay*i)
    model.optimizer = Adam(lr=lr)
    print(f"lr is {lr:.12f}" )
    model.fit(Xs,Ys, shuffle=True, validation_data=(Xst, Yst), epochs=1, batch_size=400, callbacks=[batch_callback])
    loss = buff['loss']
    val_loss = buff['val_loss']
    model.save('models/{:.09f}_{:.09f}_{:09d}_{:.12f}.h5'.format(loss,val_loss,i,init_rate*(1.0 - decay*i)))

import itertools
from scipy.io import wavfile
 
if '--predict' in sys.argv:
  model.load_weights(sorted(glob.glob('models/*.h5')).pop(0))
  Xs, Ys = np.load('Xs.npy'), np.load('Ys.npy')
  S = 5
  Yp = model.predict(Xs)
  
  Yp = Yp*32768 - 32768
  yps = np.array(list(itertools.chain(*Yp.tolist())), dtype=np.int16)   
  wavfile.write(f'yp_orig_{S}.wav', 44100, yps)
   
