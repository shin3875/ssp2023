import numpy as np
import librosa
from matplotlib import pyplot as plt
import scipy

"""
Assignment of lab 3

Student : Seunghyeon Shin
"""


# 1. getFIRFFT Function
def lp_fir_filter(input_h, window_name='hamming'):
    nfft = len(input_h-1) * 2
    order = len(input_h-1)
    h_append = (np.append(input_h, np.flip(input_h)[1:])) * (1+0j)
    h_ifft = np.fft.ifft(h_append, nfft)
    h_linear = np.append(h_ifft[len(h_ifft)-int(order/2):], h_ifft[0:int(order/2)+1])
    window = scipy.signal.get_window(window_name, order+1)
    h_windowed = h_linear*window

    return h_windowed.real


# Validation
for order in [4, 6, 14, 30, 62]:
    H = [1.0]*(order//2) + [0.0]*(order//2)
    h2 = lp_fir_filter(np.array(H))
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(h2)
    ax[0].set_title('Assign h')
    ax[1].plot(np.abs(np.fft.fft(h2, 512)))
    ax[1].set_title('Assign fft')
    plt.show()


# 2. Time-varying bandpass filter, with trapezoidal window
# requirement : 10ms shift, 3ms overlap, trapezoidal window, and time-varying bandpass filter
test_file = librosa.example('trumpet')
hop_ms = 10
frame_ms = 20
buffer_ms = 3
filter_ms = 500

test_wave, sampling = librosa.load(test_file, sr=16000)
hop_len = int((hop_ms / 1000) * sampling)
frame_len = int((frame_ms / 1000) * sampling)
buffer_len = int((buffer_ms / 1000) * sampling)
filter_shift = hop_len * int(filter_ms / hop_ms)
filter_idx = 0
filter_out = np.zeros(len(test_wave))
triangle = np.linspace(0, 1, buffer_len)
trapezoidal = np.append(triangle, np.ones(hop_len-buffer_len))
trapezoidal = np.append(trapezoidal, np.flip(triangle))
filter_hz = 1000


for i in range(int(len(test_wave)/hop_len)):
    start_idx = i * hop_len
    end_idx = (i+1) * hop_len
    filter_indicator = int(end_idx // filter_shift)
    if filter_idx == filter_indicator:
        h_array = scipy.signal.firwin(hop_len+buffer_len, cutoff=filter_hz, fs=sampling)
        if filter_hz < (sampling/2):
            filter_hz = filter_hz + 100
        filter_idx += 1

    if i == 0:
        filtered = scipy.signal.lfilter(b=h_array, a=[1], x=test_wave[start_idx:end_idx])
        buffer = test_wave[end_idx-buffer_len:end_idx]
        filter_out[start_idx:end_idx] = filtered * trapezoidal[buffer_len:]
    else:
        signal_in = np.append(buffer, test_wave[start_idx:end_idx])
        filtered = scipy.signal.lfilter(b=h_array, a=[1], x=signal_in)
        buffer = test_wave[end_idx-buffer_len:end_idx]
        filter_out[start_idx-buffer_len:end_idx] = filtered * trapezoidal + filter_out[start_idx-buffer_len:end_idx]

stft_0 = librosa.stft(test_wave, )
stft_1 = librosa.stft(filter_out, )
fig, ax = plt.subplots(1, 2)
img0 = librosa.display.specshow(librosa.amplitude_to_db(stft_0, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0], sr=sampling)
ax[0].set_title(f'Original output')
fig.colorbar(img0, ax=ax[0], format="%+2.0f dB")
img1 = librosa.display.specshow(librosa.amplitude_to_db(stft_1, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[1], sr=sampling)
ax[1].set_title(f'Filtered output')
fig.colorbar(img1, ax=ax[1], format="%+2.0f dB")

plt.show()
