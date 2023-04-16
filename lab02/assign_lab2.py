"""
Speech Signal Processing assignments

Professor : Gil-Jin Jang

Student : Seunghyeon Shin (2022325988)

"""

import numpy as np
import librosa
import librosa.display
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf

def resampling(sound, original_sr, target_sr):

  if int(target_sr/original_sr) > 0:
    ratio = int(target_sr / original_sr)
    out_data = np.zeros(len(sound)*ratio)
    for idx, sample in enumerate(out_data):
      ori_idx = int(idx / ratio)
      if int(idx % ratio) == 0:
        out_data[idx] = sound[int(idx/ratio)]
        count = 1

      elif ori_idx == len(sound):

        inter_0 = sound[ori_idx]
        inter_1 = sound[ori_idx+1]
        inter_idx = idx % ratio
        (inter_0 * (ratio - inter_idx)) / ratio
        (inter_1 * count) / ratio
        out_data[idx] = (inter_0 + inter_1) /2
      else:
        out_data[idx] = sound[ori_idx]

      count += 1
    lp_e = scipy.signal.firwin(301, cutoff=(original_sr/2), fs=target_sr, window='hamming', pass_zero='lowpass')
    out_data = scipy.signal.convolve(out_data, lp_e)

    # filtered = scipy.signal.lfilter(lp_e, 1 / ratio, out_data)
  elif (target_sr/original_sr) < 1:
    ratio = int(original_sr / target_sr)
    out_data = np.zeros(int(len(sound) / ratio))
    for idx, sample in enumerate(out_data):
      out_data[idx] = sound[int(idx*ratio)]

  return out_data


def resampler(in_data, original_sr, target_sr, file_name):
  if original_sr %target_sr == 0 or target_sr % original_sr ==0:
    out_data = resampling(in_data, original_sr, target_sr)
  else:
    b = target_sr
    a = original_sr
    while b != 0:
      r = a % b
      a = b
      b = r
    temp_sr = int(original_sr * target_sr / a)
    count = int(temp_sr / original_sr)
    out_data2 = np.zeros(len(in_data)*count)
    idx = 0
    idx_c = 0
    for i in range(len(out_data2)):
      if int(idx_c) == count:
        idx = idx + 1
        idx_c = 0
      out_data2[i] = in_data[idx]
      idx_c = idx_c + 1

    out_data = resampling(out_data2, temp_sr, target_sr)

  sf.write(f'./{file_name}_{target_sr}Hz.wav', out_data, target_sr)

  return out_data

input_file = './sh_kdigits_0_3_32000Hz.wav'
data, fs = librosa.load(input_file, sr=16000)

target_fs = 8000
test = resampler(data, original_sr=fs, target_sr=target_fs, file_name='sh_kdigits_0_3')

value = np.quantile(abs(test), 0.8)
binary_mask = (abs(test) > value)
test = test * binary_mask

stft_0 = librosa.stft(data, )
stft_1 = librosa.stft(test, )
fig, ax = plt.subplots(1, 2)
img0 = librosa.display.specshow(librosa.amplitude_to_db(stft_0, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax[0], sr=fs)
ax[0].set_title(f'Original output {fs}Hz')
fig.colorbar(img0, ax=ax[0], format="%+2.0f dB")
img1 = librosa.display.specshow(librosa.amplitude_to_db(stft_1, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax[1], sr=target_fs)
ax[1].set_title(f'Resample output {target_fs}Hz')
fig.colorbar(img1, ax=ax[1], format="%+2.0f dB")

plt.show()