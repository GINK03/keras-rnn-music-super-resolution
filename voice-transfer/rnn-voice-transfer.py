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
from keras.losses               import mean_squared_error
from pathlib import Path
import numpy as np
import random
import sys
import pickle
import gzip
import glob
import copy
import os
import re
import time

width = 20
input1      = Input(shape=(width, 1))
x           = TD(Dense(1000, activation='linear'))(input1)
x           = GRU(300, activation='linear', recurrent_activation='linear', return_sequences=True)(x)
x           = Bi(GRU(300, activation='linear', recurrent_activation='linear', return_sequences=True))(x)
x           = TD(Dense(3000, activation='linear'))(x)
x           = TD(Dense(3000, activation='linear'))(x)
output      = TD(Dense(1, activation='linear'))(x)

model       = Model(input1, output)
model.compile(RMSprop(lr=0.0001, decay=0.03), loss='mae')


buff = None
now  = time.strftime("%H_%M_%S")
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
batch_callback = LambdaCallback(on_epoch_end=lambda batch,logs: callback(batch,logs) )


if '--train' in sys.argv:
  if '--resume' in sys.argv:
    model.load_weights(sorted(glob.glob('./models/*.h5')).pop(0))
  paths = [f'{path}' for path in Path('./blob/').glob('*') if f'{path}'.split('/').pop()[0] != '7'] 
  #print(paths)
  decay = 0.0001
  init_rate =  0.0005
  
  for i in range(1000):
    samples = random.sample(paths, 1)
    Xs,Ys = None, None
    for sample in samples:
      Xs, Ys = pickle.loads(gzip.decompress(open(sample, 'rb').read()))
      lr = init_rate*(1.0 - decay*i)
      model.optimizer = Adam(lr=lr)
      print(f"lr is {lr:.12f}" )
      model.fit(x=Xs, y=Ys, shuffle=True, validation_split=0.1, epochs=100, batch_size=240, callbacks=[batch_callback])
      loss = buff['loss']
      val_loss = buff['val_loss']

    if i%50 == 0:
      model.save('models/{:.09f}_{:.09f}_{:09d}_{:.12f}.h5'.format(loss,val_loss,i,init_rate*(1.0 - decay*i)))

import itertools
from scipy.io import wavfile

if '--predict' in sys.argv:
  model.load_weights(sorted(glob.glob('models/0.228663650_0.097335294_000000900_0.000010000000.h5')).pop(0))
  Xs, Ys = pickle.loads(gzip.decompress(open('predicts/d5c8df5e9071b6054e735473d92016b435dc73831c08ab51626b0826dd6b2767.pkl', 'rb').read()))

  S = 2
  if '--baseline' in sys.argv: 
    Xs = Xs*10000
    xs = np.array(list(itertools.chain(*Xs.tolist())), dtype=np.int16)   
    wavfile.write(f'xs_orig_{S}.wav', 4410*10, xs)
    
    Ys = Ys*10000
    for y in Ys.tolist():
      print(y)
    ys = np.array(list(itertools.chain(*Ys.tolist())), dtype=np.int16)   
    wavfile.write(f'ys_orig_{S}.wav', 4410*10, ys)
  else: 
    Yp = model.predict(Xs)
    Yp = Yp*10000
    for y in Yp.tolist():
      print(y)
    yps = np.array(list(itertools.chain(*Yp.tolist())), dtype=np.int16)   
    wavfile.write(f'yp_orig_{S}.wav', 4410*10, yps)
   
