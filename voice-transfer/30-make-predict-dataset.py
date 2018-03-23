
from pathlib import Path

from scipy.io import wavfile

import numpy as np

import re

import gzip, pickle

import hashlib
for name in Path('../../sda/tmp/').glob('*'):
  num = re.search('\d{1,}', str(name)).group(0) 
  kyoko = np.array(wavfile.read(f'../../sda/tmp/Kyoko_{num}.wav')[1], dtype=np.int16)
  otoya = np.array(wavfile.read(f'../../sda/tmp/Otoya_{num}.wav')[1], dtype=np.int16)
  
  Xs = otoya.tolist()
  Ys = kyoko.tolist() 

  sources, targets = [], []
  for i in range(0,len(Xs),250): 
    source = Xs[i:i+250] 
    target = Ys[i:i+250] 
    if len(source) == 250 and len(target) == 250:
      sources.append( source )
      targets.append( target )
  length = len(sources)
  sources = np.array(sources, dtype=float).reshape((length, 250,1))/10000
  targets = np.array(targets, dtype=float).reshape((length, 250,1))/10000
  data = gzip.compress(pickle.dumps( (sources, targets) ))
  ha = hashlib.sha256(data).hexdigest()
  open(f'predicts/{ha}.pkl', 'wb').write( data ) 
  
  break
