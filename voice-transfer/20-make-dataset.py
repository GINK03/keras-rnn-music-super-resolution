import numpy as np
from scipy.io import wavfile 

import pickle
import gzip
import time
import random

from pathlib import Path
import re
for name in Path('../../sda/tmp/').glob('Kyoko_*'):
  print(name)
  num = re.search('\d{1,}', str(name)).group(0)
  print(num)
  kyoko = np.array(wavfile.read(f'../../sda/tmp/Kyoko_{num}.wav')[1], dtype=np.int16)
  print( wavfile.read(f'../../sda/tmp/Kyoko_{num}.wav') )
  otoya = np.array(wavfile.read(f'../../sda/tmp/Otoya_{num}.wav')[1], dtype=np.int16)

  Xs = otoya.tolist()
  Ys = kyoko.tolist()
  m = max(Xs) 
  print('max', m)
  sources, targets = [], []
  print(len(Xs))
  for i in range(0,len(Xs),20):
    source = Xs[i:i+250]
    target = Ys[i:i+250]
    if len(source) == 250 and len(target) == 250:
      sources.append( source )
      targets.append( target )

  length = len(sources)
  sources = np.array(sources, dtype=float).reshape((length, 250,1))/25000
  targets = np.array(targets, dtype=float).reshape((length, 250,1))/25000
  print(length, sources.shape)
  open(f'dataset/{num}.pkl', 'wb').write( gzip.compress(pickle.dumps( (sources, targets) )) )

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
