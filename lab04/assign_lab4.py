import numpy as np
import librosa
from matplotlib import pyplot as plt
import scipy


"""
Assignment of lab 3

Student : Seunghyeon Shin
"""

# Basic setting
sample_file = 'kdigits7_0.wav'
noise_time = 0.3
snr_list = [0, 10]
voice, sampling = librosa.load(sample_file, sr=16000)


def lp_fir_filter(input_h, window_name='hamming'):
    nfft = len(input_h-1) * 2
    order = len(input_h-1)
    h_append = (np.append(input_h, np.flip(input_h)[1:])) * (1+0j)
    h_ifft = np.fft.ifft(h_append, nfft)
    h_linear = np.append(h_ifft[len(h_ifft)-int(order/2):], h_ifft[0:int(order/2)+1])
    window = scipy.signal.get_window(window_name, order)
    h_windowed = h_linear*window

    return h_windowed.real

# Noise setting
def noise_insert(input_signal, snr, sr, noise_sec):

    noise_len = max(noise_sec*sr, len(input_signal))
    awgn_len = int(sr * noise_sec)
    awgn = np.random.normal(0, 1, noise_len)
    awgn = awgn / np.max(abs(awgn))
    awgn_rms = np.sum(np.sqrt(awgn*awgn)) / len(awgn)
    signal_rms = np.sum(np.sqrt(input_signal*input_signal)) / len(input_signal)

    snr_ratio = 10 * np.log10(signal_rms / awgn_rms)
    snr_adj = snr_ratio - snr
    ratio = 10 ** (snr_adj / 10)
    awgn_adj = awgn * ratio
    output = np.append(np.append(awgn_adj[:awgn_len], input_signal + awgn_adj[:len(input_signal)]), awgn_adj[:awgn_len])

    return output


noise_voice = noise_insert(voice, snr=10, sr=sampling, noise_sec=noise_time)


# estimate autocorrelation and crosscorrelation
# nt : sample of period, nf = sr * frame(20ms), ns = sample shift, time(10ms), hno = nfft/2 +1
def correlation_calc(target_signal, start_time, end_time, sr, frame_sec, sample_shift=0.02):
    # fft per input frame, and multiply conjugate, and sum of real part
    frame_len = int(frame_sec * sr)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    signal_clip = target_signal[start_sample:end_sample]
    sample_shift = sr*sample_shift
    nfft = int(2**(np.ceil(np.log2(frame_len))))
    h_num = int(nfft/2 + 1)
    sxx_sum = np.zeros(h_num)
    for i in range(int((end_time-start_time)*sr)):
        period_len = len(signal_clip)
        frame_count = int((period_len-frame_len) // sample_shift+1)
        sxx = np.zeros(h_num)
        for j in range(frame_count):
            start_idx = int(j*sample_shift+start_sample)
            fft_x = np.fft.fft(signal_clip[start_idx:int(start_idx+sample_shift)], nfft)
            x_sq = fft_x * (fft_x*(1-j))
            sxx += x_sq[:int(nfft/2+1)].real
        sxx = sxx / frame_count
        sxx_sum = sxx_sum + sxx
    return sxx_sum / (j+1)


corr_all = correlation_calc(noise_voice, 0.0, len(noise_voice)/sampling, sr=sampling, frame_sec=0.04)
corr_noise = correlation_calc(noise_voice, 0.0, 0.5, sr=sampling, frame_sec=0.04)
corr_voice = correlation_calc(noise_voice, 0.5, len(noise_voice)/sampling, sr=sampling, frame_sec=0.04)


nfft = int(2**(np.ceil(np.log2(0.04*sampling))))
frame_sample = int(nfft / 2 + 1)
frame_total = int(len(noise_voice)/frame_sample + 1)
rms = []

filtered = np.zeros(len(noise_voice))
for i in range(frame_total):
    start_point = int(i * frame_sample)
    end_point = int((i+1) * frame_sample)
    if end_point > len(noise_voice):
        end_point = len(noise_voice) - 1
    voice_in = noise_voice[start_point:end_point]
    fft_voice = np.fft.fft(voice_in, nfft)
    voice_sq = fft_voice * (fft_voice * complex(1, -1))
    voice_sum = voice_sq[:int(nfft/2+1)].real
    h = (voice_sum - corr_noise) / voice_sum
    h_out = lp_fir_filter(h)
    filtered[start_point:end_point] = scipy.signal.lfilter(b=h_out, a=[1], x=voice_in)

    signal_rms = np.sum(np.sqrt(voice_in*voice_in)) / len(voice_in)
    rms = np.append(rms, signal_rms)


# Endpoint detection - energy based
rms_mean = np.mean(rms)
rms_bool = (rms > rms_mean)
idx = np.where(rms_bool == True)
start_idx = idx[0] - 1
end_idx = idx[-1] + 1
voice_idx = (start_idx * frame_sample) / sampling

corr_noise = correlation_calc(noise_voice, 0.0, voice_idx, sr=sampling, frame_sec=0.04)

filtered_rev = np.zeros(len(noise_voice))
for i in range(frame_total):
    start_point = int(i * frame_sample)
    end_point = int((i+1) * frame_sample)
    if end_point > len(noise_voice):
        end_point = len(noise_voice) - 1
    voice_in = noise_voice[start_point:end_point]
    fft_voice = np.fft.fft(voice_in, nfft)
    voice_sq = fft_voice * (fft_voice * complex(1, -1))
    voice_sum = voice_sq[:int(nfft/2+1)].real
    h = (voice_sum - corr_noise) / voice_sum
    h_out = lp_fir_filter(h)
    filtered_rev[start_point:end_point] = scipy.signal.lfilter(b=h_out, a=[1], x=voice_in)
