from keras.layers               import Input, Dense, GRU, LSTM, CuDNNLSTM, RepeatVector
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

input_tensor1 = Input(shape=(250, 1))
x1          = Bi(CuDNNLSTM(300, return_sequences=True))(input_tensor1)
x           = Dense(1000, activation='relu')(x1)
x           = Bi(CuDNNLSTM(300, return_sequences=True))(x)
x           = TD(Dense(500, activation='linear'))(x)
decoded     = TD(Dense(1, activation='linear'))(x)

model       = Model(input_tensor1, decoded)
model.compile(RMSprop(lr=0.0001, decay=0.03), loss='mae')

buff = None
now  = time.strftime("%H_%M_%S")
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
batch_callback = LambdaCallback(on_epoch_end=lambda batch,logs: callback(batch,logs) )


if '--train' in sys.argv:
  Xs, Ys = pickle.load(open('dataset.pkl', 'rb'))
  if '--resume' in sys.argv:
    model.load_weights(sorted(glob.glob('./models/*.h5')).pop(0))
  print(Xs.shape)
  decay = 0.03
  init_rate =  0.00009
  for i in range(33):
    lr = init_rate*(1.0 - decay*i)
    model.optimizer = Adam(lr=lr)
    print(f"lr is {lr:.12f}" )
    model.fit(Xs,Ys, shuffle=True, validation_split=0.1, epochs=1, batch_size=240, callbacks=[batch_callback])
    loss = buff['loss']
    val_loss = buff['val_loss']
    model.save('models/{:.09f}_{:.09f}_{:09d}_{:.12f}.h5'.format(loss,val_loss,i,init_rate*(1.0 - decay*i)))

import itertools
from scipy.io import wavfile

if '--predict' in sys.argv:
  model.load_weights(sorted(glob.glob('models/*.h5')).pop(0))
  Xs, Ys = pickle.load(open('predict.pkl', 'rb'))

  S = 2
  if '--baseline' in sys.argv: 
    Xs = Xs*32766
    xs = np.array(list(itertools.chain(*Xs.tolist())), dtype=np.int16)   
    wavfile.write(f'xs_orig_{S}.wav', 4410*4, xs)
    
    Ys = Ys*32767
    ys = np.array(list(itertools.chain(*Ys.tolist())), dtype=np.int16)   
    wavfile.write(f'ys_orig_{S}.wav', 4410*4, ys)
  else: 
    Yp = model.predict(Xs)
    
    Yp = Yp*32767
    yps = np.array(list(itertools.chain(*Yp.tolist())), dtype=np.int16)   
    wavfile.write(f'yp_orig_{S}.wav', 4410*4, yps)
   
