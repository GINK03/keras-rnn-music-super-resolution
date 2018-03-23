import numpy as np
from scipy.io import wavfile 

import pickle
import gzip
import time
import random
import hashlib
from pathlib import Path
import re
for enum, name in enumerate(Path('../../sda/tmp/').glob('Kyoko_*')):
  print(name)
  num = re.search('\d{1,}', str(name)).group(0)
  print(num)
  otoya = np.array(wavfile.read(f'../../sda/tmp/Otoya_{num}.wav')[1], dtype=np.int16)
  kyoko = np.array(wavfile.read(f'../../sda/tmp/Kyoko_{num}.wav')[1], dtype=np.int16)

  Otoya = otoya.tolist()
  Kyoko = kyoko.tolist()

  Xs, Ys = [], []
  print(len(Xs))
  width = 20
  for i in range(0,len(Otoya),20):
    x = Otoya[i:i+width]
    y = Kyoko[i:i+width]
    if len(x) == width and len(y) == width:
      Xs.append( x )
      Ys.append( y )
    
    if len(Xs) >= 25000:
      length = len(Xs)
      Xs = np.array(Xs, dtype=float).reshape((length, width,1))/10000
      Ys = np.array(Ys, dtype=float).reshape((length, width,1))/10000
      print(length, Xs.shape)
      data = gzip.compress(pickle.dumps( (Xs, Ys) ) )
      ha = hashlib.sha256(data).hexdigest()
      open(f'blob/{ha}.pkl', 'wb').write( data )
      Xs, Ys = [], []
  if enum >= 10:
    break


'''
sources, targets = [], []
for i in range(0,len(Xs),50):
  source = sum(Xs[i:i+50],[])
  target = sum(Ys[i:i+50],[]) 
  if len(source) == 250 and len(target) == 250:
    sources.append( source )
    targets.append( target )

length = len(sources)
sources = np.array(sources, dtype=float).reshape((length, 250,1))/32767

targets = np.array(targets, dtype=float).reshape((length, 250,1))/32767
print(sources.shape)
open('predict.pkl', 'wb').write( pickle.dumps( (sources, targets) ) )
'''
