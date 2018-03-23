from scipy.io import wavfile
import numpy as np
import pickle

maxlen = 8979450
minlen = int(maxlen*0.8)
wav = wavfile.read('waves/reworu-wave.wav')
org = np.array(wav[1])[:,0][minlen:maxlen]

dig = wavfile.read('degradation.wav')
dig = np.array(dig[1])[minlen:maxlen]

prd = wavfile.read('yp_orig_5.wav')
prd = np.array(prd[1])[minlen:maxlen]
print( np.abs(org - org).sum()/len(org) )
print( np.abs(org - dig).sum()/len(org) )
print( np.abs(org - prd).sum()/len(org) )
