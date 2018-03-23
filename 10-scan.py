from scipy.io import wavfile
import numpy as np
import pickle
wav = wavfile.read('waves/reworu-wave.wav')
#print(wav)
# sample rate:44100
# 片方の音源に限定
wav = np.array(wav[1])[:,0]
print('wave shape', wav.shape)
print('wave min', np.min(wav))
print('wave max', np.max(wav))

# 16bit表現なので、0 ~ 65535に収まるはずであり、np.minが-32768なので足し合わせる
nextwav = []
Ys = []
for index, w in enumerate(wav.tolist()):
  #if w != 0 :
  nextwav.append(w)
  Ys.append( (w+32768)/32768 )
nextwav = np.array(nextwav,dtype=np.int16 )
print(nextwav.shape)
wavfile.write('origin.wav', 44100, nextwav)
#sys.exit()

nextwav = []
Xs = []
for index, w in enumerate(wav.tolist()):
  if index%2 == 0:
    wabs = w + 32768
    wabs = f'{wabs:016b}'
    Xs.append( wabs )
    Xs.append( wabs )
    nextwav.append(w)
    nextwav.append(w)
nextwav = np.array(nextwav,dtype=np.int16 )
print(nextwav.shape)
wavfile.write('degradation.wav', 44100//2, nextwav)

open('xs_ys.pkl', 'wb').write( pickle.dumps( (Xs, Ys) ) )

