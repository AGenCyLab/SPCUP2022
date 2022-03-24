
import librosa
from methods.cqcc import cqcc
import numpy as np


filename = 'D18_1000001.wav'
sr = 16000
# load input signal
[x, fs] = librosa.load(filename, sr=sr)

# parameters
B = 96
fmax = fs/2
fmin = fmax/(2**9)
d = 16
cf = 19
ZsdD = 'ZsdD'

# compute cqcc
cqcc_feature = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD) 


print(cqcc_feature.shape)
print(cqcc_feature[0:10, 0])
