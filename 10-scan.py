from scipy.io import wavfile
import numpy as np

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
for index, w in enumerate(wav.tolist()):
  if w != 0 :
    wabs = w + 32768
    wabs = f'{wabs:016b}'
    #print(index, w, wabs)
    nextwav.append(w)
nextwav = np.array(nextwav,dtype=np.int16 )
print(nextwav.shape)
wavfile.write('origin.wav', 44100, nextwav)
sys.exit()

nextwav = []
for index, w in enumerate(wav.tolist()):
  if index%2 == 0:
    #print(index, w)
    nextwav.append(w)
nextwav = np.array(nextwav,dtype=np.int16 )
print(nextwav.shape)
wavfile.write('degradation.wav', 44100//2, nextwav)
