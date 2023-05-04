import os
import glob
import pickle
import scipy
import librosa
import librosa.display
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from hmmlearn import hmm


# VAD
# input : single sound
# output : single / cutted sample, activity time
def vad(sound, sampling, silent_interval=0.2):

    mel = librosa.feature.melspectrogram(y=sound, sr=sampling, n_mels=128, fmax=8000)
    mfcc = librosa.feature.mfcc(y=sound, sr=sampling, n_mfcc=13)

    mfcc_mean = (np.mean(mfcc, axis=-1))
    mfcc_list = sorted((np.mean(mfcc, axis=-1)))
    mel_mean = np.mean(mel, axis=-1)
    mel_list = sorted((np.mean(mel, axis=-1)))

    time_len_mfcc = len(sound) / len(mfcc[1]) * (1 / sampling)
    time_bin_mel = len(sound) / len(mel[1]) * (1 / sampling)

    silent_mfcc_bin = int(silent_interval / time_len_mfcc)
    silent_mel_bin = int(silent_interval / time_bin_mel)

    activation_mfcc = np.mean([mfcc[int(np.where(mfcc_mean == mfcc_list[-1])[0]), :],
                              mfcc[int(np.where(mfcc_mean == mfcc_list[-2])[0]), :]] , axis=0)
    activation_mel = np.mean([mel[int(np.where(mel_mean == mel_list[-1])[0]), :],
                              mel[int(np.where(mel_mean == mel_list[-2])[0]), :]] , axis=0)

    mean_mfcc = np.mean(activation_mfcc)
    mean_mel = np.mean(activation_mel)
    thr_mfcc = activation_mfcc > mean_mfcc
    thr_mel = activation_mel > mean_mel

    act_list = []
    act_idx = 0
    for idx, tf_bin in enumerate(thr_mfcc):
        if tf_bin and (idx - act_idx) > silent_mfcc_bin:
            act_list.append(idx)
            act_idx = idx
        elif tf_bin and (act_idx - idx) <= silent_mfcc_bin:
            act_idx = idx

    split_list = []
    split_signal = []
    noise_signal = []
    for i in range(len(act_list)):
        start_time = (act_list[i] * time_len_mfcc) - silent_interval
        if i == 0:
            noise_signal.append(sound[:int(start_time * sampling)])
        else:
            noise_signal.append(sound[int(end_time * sampling):int(start_time * sampling)])

        if i == len(act_list) - 1:
            end_time = len(act_list) * time_len_mfcc - (silent_interval * 2)

        else:
            end_time = (act_list[i+1] * time_len_mfcc) - (silent_interval * 3)
        time_list = [start_time, end_time]
        split_list.append(time_list)
        split_signal.append(sound[int(start_time*sampling):int(end_time*sampling)])

    return split_signal, split_list, noise_signal


# Revision of wiener filter code
# Original code : https://github.com/raphaelvdumas/noise-reduction
#
class Wiener:
    """
    Class made for wiener filtering based on the article "Improved Signal-to-Noise Ratio Estimation for Speech
    Enhancement".
    Reference :
        Cyril Plapous, Claude Marro, Pascal Scalart. Improved Signal-to-Noise Ratio Estimation for Speech
        Enhancement. IEEE Transactions on Audio, Speech and Language Processing, Institute of Electrical
        and Electronics Engineers, 2006.

    modification : input type, channel

    """

    def __init__(self, audio, sampling, noise_t):
        """
        Input :
            WAV_FILE
            T_NOISE : float, Time in seconds /!\ Only works if stationnary noise is at the beginning of x /!\

        """
        # Constants are defined here
        self.WAV_FILE, self.T_NOISE = audio, noise_t
        self.Fs, self.x = sampling, audio
        self.NFFT, self.SHIFT, self.T_NOISE = 2 ** 10, 0.5, noise_t
        self.FRAME = int(0.02 * sampling)  # Frame of 20 ms

        # Computes the offset and number of frames for overlapp - add method.
        self.OFFSET = int(self.SHIFT * self.FRAME)

        # Hanning window and its energy Ew
        self.WINDOW = scipy.signal.windows.hann(self.FRAME)
        self.EW = np.sum(self.WINDOW)

        self.frames = np.arange((len(audio) - self.FRAME) // self.OFFSET + 1)
        # Evaluating noise psd with n_noise
        self.Sbb = self.welchs_periodogram()

    @staticmethod
    def a_posteriori_gain(snr):
        """
        Function that computes the a posteriori gain G of Wiener filtering.

            Input :
                SNR : 1D np.array, Signal to Noise Ratio
            Output :
                G : 1D np.array, gain G of Wiener filtering

        """
        gain = (snr - 1) / snr
        return gain

    @staticmethod
    def a_priori_gain(snr):
        """
        Function that computes the a priori gain G of Wiener filtering.

            Input :
                SNR : 1D np.array, Signal to Noise Ratio
            Output :
                G : 1D np.array, gain G of Wiener filtering

        """
        gain = snr / (snr + 1)
        return gain

    def welchs_periodogram(self):
        """
        Estimation of the Power Spectral Density (Sbb) of the stationary noise
        with Welch's periodogram given prior knowledge of n_noise points where
        speech is absent.

            Output :
                Sbb : 1D np.array, Power Spectral Density of stationary noise

        """
        # Initialising Sbb
        Sbb = np.zeros(self.NFFT)

        noise_on = self.T_NOISE[0] * self.Fs
        noise_off = self.T_NOISE[1] * self.Fs
        noise_frames = int(((noise_off - noise_on) - self.FRAME) // self.OFFSET +1)
        offset = self.OFFSET
        # Number of frames used for the noise
        # noise_frames = np.arange(((self.N_NOISE[1] - self.N_NOISE[0]) - self.FRAME) // self.OFFSET + 1)
        for frame in range(noise_frames):
            i_min, i_max = frame * offset + noise_on, frame * offset + self.FRAME + noise_on
            x_framed = self.x[i_min:i_max] * self.WINDOW
            X_framed = fft(x_framed, self.NFFT)
            Sbb[:] = frame * Sbb[:] / (frame + 1) + np.abs(X_framed) ** 2 / (frame + 1)
        return Sbb

    def moving_average(self):
        # Initialising Sbb
        Sbb = np.zeros(self.NFFT)
        # Number of frames used for the noise
        len_noise = (self.T_NOISE[1]*self.FS - self.FRAME +1)
        for frame in range(len_noise):
            x_framed = self.x[frame:frame + self.FRAME] * self.WINDOW
            X_framed = fft(x_framed, self.NFFT)
            Sbb[:] += np.abs(X_framed) ** 2
        return Sbb / len(len_noise)

    def wiener(self):
        """
        Function that returns the estimated speech signal using overlapp - add method
        by applying a Wiener Filter on each frame to the noised input signal.

            Output :
                s_est : 1D np.array, Estimated speech signal

        """
        # Initialising estimated signal s_est
        s_est = np.zeros(self.x.shape)

        for frame in self.frames:
            ############# Initialising Frame ###################################
            # Temporal framing with a Hanning window
            i_min, i_max = frame * self.OFFSET, frame * self.OFFSET + self.FRAME
            x_framed = self.x[i_min:i_max] * self.WINDOW

            # Zero padding x_framed
            X_framed = fft(x_framed, self.NFFT)

            ############# Wiener Filter ########################################
            # Apply a priori wiener gains G to X_framed to get output S
            SNR_post = (np.abs(X_framed) ** 2 / self.EW) / self.Sbb[:]
            G = Wiener.a_priori_gain(SNR_post)
            S = X_framed * G

            ############# Temporal estimated Signal ############################
            # Estimated signals at each frame normalized by the shift value
            temp_s_est = np.real(ifft(S)) * self.SHIFT
            s_est[i_min:i_max] += temp_s_est[:self.FRAME]  # Truncating zero padding

        return s_est / abs(s_est).max()


# revision of Hmm model by Gil-Jin Jang
def init_by_bakis(inumstates, ibakisLevel):

    prob_prior = np.zeros(inumstates)
    prob_prior[0: ibakisLevel - 1] = 1/float((ibakisLevel - 1))
    transmat_prior = get_transmat_prior(inumstates, ibakisLevel)

    return prob_prior, transmat_prior


def get_transmat_prior(inumstates, ibakisLevel):
    transmat_prior = (1 / float(ibakisLevel)) * np.eye(inumstates)

    for i in range(inumstates - (ibakisLevel - 1)):
        for j in range(ibakisLevel - 1):
            transmat_prior[i, i + j + 1] = 1. / ibakisLevel

    for i in range(inumstates - ibakisLevel + 1, inumstates):
        for j in range(inumstates - i - j):
            transmat_prior[i, i + j] = 1. / (inumstates - i)

    return transmat_prior


class GMMHMmodel:
    def __init__(self, class_idx, label, n_mfcc, n_comp, n_mix, transmat_p, start_p, cov_type, n_iter):

        self.traindata = np.zeros((0, n_mfcc))
        self.class_idx = class_idx
        self.label = label
        self.model = hmm.GMMHMM(
            n_components=n_comp, n_mix=n_mix, transmat_prior=transmat_p,
            startprob_prior=start_p, covariance_type=cov_type, n_iter=n_iter)


def hmm_model(train_data, train_label, val_data, val_label, test_data, test_label):

    n_hmmstat = 3  # number of states
    # m_num_of_HMMStates = 6  # number of states
    n_mix = 2  # number of mixtures for each hidden state
    cov_type = 'diag'  # covariance type
    m_n_iter = 10  # number of iterations
    bakis_l = 2
    n_mfcc = len(train_data[0])

    start_prob, trans_prob = init_by_bakis(n_hmmstat, bakis_l)

    spoken = list(set(train_label))

    for i in range(len(train_data)):
        train_data[i] = np.transpose(train_data[i])

    for i in range(len(val_data)):
        val_data[i] = np.transpose(val_data[i])

    for i in range(len(test_data)):
        test_data[i] = np.transpose(test_data[i])

    speechmodels = [None] * len(spoken)

    for idx, word in enumerate(spoken):
        speechmodels[word] = GMMHMmodel(
            word, idx, n_mfcc, n_hmmstat, n_mix, trans_prob, start_prob, cov_type, m_n_iter)

    for i in range(0, len(train_data)):
         for j in range(0, len(speechmodels)):
             if int(speechmodels[j].class_idx) == int(train_label[i]):
                speechmodels[j].traindata = np.concatenate((speechmodels[j].traindata, train_data[i]))

    for speechmodel in speechmodels:
        speechmodel.model.fit(speechmodel.traindata)

    test_pred = []
    val_pred = []

    for val_idx in range(len(val_data)):
        score = []
        for speechmodel in speechmodels:
            score.append(speechmodel.model.score(val_data[val_idx]))

        id = score.index(max(score))
        val_pred.append(speechmodels[id].class_idx)


    for test_idx in range(len(test_data)):
        score = []
        for speechmodel in speechmodels:
            score.append(speechmodel.model.score(test_data[test_idx]))

        id = score.index(max(score))
        test_pred.append(speechmodels[id].class_idx)

    count_test = 0
    count_val = 0

    for index, single_p in enumerate(val_pred):
        if single_p == val_label[index]:
            count_val += 1

    for index, single_p in enumerate(test_pred):
        if single_p == test_label[index]:
            count_test += 1

    accuracy_val = 100*count_val/float(len(val_label))
    accuracy_test = 100*count_test/float(len(test_label))

    print(f"\nAccuracy at validation data: {accuracy_val}")
    print(f"\nAccuracy at test data: {accuracy_test}\n")


def dataset_generation(tag='extracted_feature', base_dir='./dataset/', n_mfcc=40, normalize=True):
    query = os.path.abspath("{base}/*".format(base=base_dir))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    output_dir = base_dir + tag

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for target_digits in range(10):
        features = []
        noisy_feature = []
        for i, target_dir in enumerate(dirs):
            if 'train' in target_dir:
                mode = 'train'
            elif 'val' in target_dir:
                mode = 'val'
            elif 'test' in target_dir:
                mode = 'test'
            else:
                mode = None

            dirs_in = os.listdir(target_dir + '/')
            if mode == 'train':
                for j, speaker_dir in enumerate(dirs_in):
                    dirs_digits = os.listdir((target_dir + '/' + speaker_dir))

                    for k, digits in enumerate(dirs_digits):
                        if digits == f'{target_digits}':
                            directory = target_dir + '/' + speaker_dir + '/' + digits
                            if os.path.isdir(directory):
                                files = os.path.abspath(
                                    "{dir_name}/*.{ext}".format(dir_name=directory, ext="wav"))
                                file_paths = sorted(glob.glob(files))

                                for idx, files in enumerate(file_paths):
                                    sound, sampling = librosa.load(files, sr=16000)
                                    if normalize:
                                        sound = librosa.util.normalize(sound)
                                    mfcc = librosa.feature.mfcc(y=sound, sr=sampling, n_mfcc=n_mfcc)
                                    features.append(mfcc)
            elif mode == 'val':
                for j, snr_dirs in enumerate(dirs_in):

                    dirs_snrs = os.listdir((target_dir + '/' + snr_dirs))
                    for k, speaker_dir in enumerate(dirs_snrs):
                        dirs_digits = os.listdir((target_dir + '/' + snr_dirs + '/' + speaker_dir))

                        for l, digits in enumerate(dirs_digits):
                            if digits == f'{target_digits}':
                                directory = target_dir + '/' + snr_dirs + '/' + speaker_dir + '/' + digits
                                if os.path.isdir(directory):
                                    files = os.path.abspath(
                                        "{dir_name}/*.{ext}".format(dir_name=directory, ext="wav"))
                                    file_paths = sorted(glob.glob(files))

                                    for idx, files in enumerate(file_paths):
                                        sound, sampling = librosa.load(files, sr=16000)
                                        if normalize:
                                            sound = librosa.util.normalize(sound)
                                        mfcc = librosa.feature.mfcc(y=sound, sr=sampling, n_mfcc=n_mfcc)
                                        features.append(mfcc)
                                        if 'SNR' in snr_dirs and mode == 'val':
                                            noisy_feature.append(mfcc)

            name = f"{output_dir}/feature_{mode}_{target_digits}.pkl"
            with open(file=name, mode='wb') as f:
                pickle.dump(features, f)

    for i, target_dir in enumerate(dirs):
        if 'test' in target_dir:
            mode = 'test'
            fe_0, fe_1, fe_2, fe_3, fe_4, fe_5, fe_6, fe_7, fe_8, fe_9 = [], [], [], [], [], [], [], [], [], []

            for j, snr_dirs in enumerate(dirs_in):

                dirs_snrs = os.listdir((target_dir + '/' + snr_dirs))
                for k, speaker_dir in enumerate(dirs_snrs):
                    dirs_digits = os.listdir((target_dir + '/' + snr_dirs + '/' + speaker_dir))

                    for l, files in enumerate(dirs_digits):
                        if f'{target_digits}' in files:
                            file_dir = target_dir + '/' + snr_dirs + '/' + speaker_dir + '/' + files

                            sound, sampling = librosa.load(file_dir, sr=16000)
                            if normalize:
                                sound = librosa.util.normalize(sound)
                            split_signal, split_list, noise_signal = vad(sound, sampling)

                            noise_in = [0, len(noise_signal[0]) / sampling]
                            filter_class = Wiener(sound, sampling, noise_in)
                            sound_nr = filter_class.wiener()

                            split_signal, split_list, noise_signal = vad(sound_nr, sampling)

                            for idx in range(len(split_signal)):
                                mfcc = librosa.feature.mfcc(y=split_signal[idx], sr=sampling, n_mfcc=n_mfcc)
                                if idx == 0:
                                    fe_1.append(mfcc)
                                elif idx == 1:
                                    fe_2.append(mfcc)
                                elif idx == 2:
                                    fe_3.append(mfcc)
                                elif idx == 3:
                                    fe_4.append(mfcc)
                                elif idx == 4:
                                    fe_5.append(mfcc)
                                elif idx == 5:
                                    fe_6.append(mfcc)
                                elif idx == 6:
                                    fe_7.append(mfcc)
                                elif idx == 7:
                                    fe_8.append(mfcc)
                                elif idx == 8:
                                    fe_9.append(mfcc)
                                elif idx == 9:
                                    fe_0.append(mfcc)

    fe_list = [fe_0, fe_1, fe_2, fe_3, fe_4, fe_5, fe_6, fe_7, fe_8, fe_9]
    for idx, f_list in enumerate(fe_list):

        name = f"{output_dir}/feature_{mode}_{idx}.pkl"
        with open(file=name, mode='wb') as f:
            pickle.dump(f_list, f)

