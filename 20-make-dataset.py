import numpy as np
from scipy.io import wavfile 

import pickle
import time
import random

import sys

Xs, Ys = pickle.loads(open('xs_ys.pkl','rb').read())

cross = list(zip(Xs, Ys))

Xsa, Ysa = [], []
for i in range(0, len(cross), 50):
  _Xs, _Ys = [], []
  for x, y in cross[i:i+50]:
    #print( x, y )
    nx = np.zeros(16)
    for index, _x in enumerate(list(x)):
      nx[index] = int(_x)
    _Xs.append( nx )    
    _Ys.append( y )
  try:
    _Xs = np.array(_Xs).reshape( (50, 16) ) 
    _Ys = np.array(_Ys).reshape( (50, ) )
  except Exception as ex:
    print(ex)
    continue
  #print(_Xs.shape, _Ys.shape)
  Xsa.append( _Xs )
  Ysa.append( _Ys )

Xsa, Ysa = map(np.array, [Xsa, Ysa])
print(Xsa.shape, Ysa.shape)
np.save('Xs', Xsa)
np.save('Ys', Ysa)
