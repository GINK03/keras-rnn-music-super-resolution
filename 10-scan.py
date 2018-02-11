from scipy.io import wavfile
import numpy as np

wav = wavfile.read('reworu-wave.wav')
print(wav)
# sample rate:44100
# 片方の音源に限定
wav = np.array(wav[1])[:,0]
print(wav.shape)

nextwav = []
for index, w in enumerate(wav.tolist()):
  if index%2 == 0:
    #print(index, w)
    nextwav.append(w)
nextwav = np.array(nextwav,dtype=np.int16 )
print(nextwav.shape)
wavfile.write('origin.wav', 44100//2, nextwav)

nextwav = []
for index, w in enumerate(wav.tolist()):
  if index%10 == 0:
    #print(index, w)
    nextwav.append(w)
nextwav = np.array(nextwav,dtype=np.int16 )
print(nextwav.shape)
wavfile.write('degradation.wav', 44100//10, nextwav)
