import sys
import numpy as np
import librosa as lb
from scipy import misc
import matplotlib.pyplot as plt


SAMPLE_FREQ = 12000
N_FFT = 512
N_MELS = 96
N_OVERLAP = 256
DURA = 29.12


def log_scale_melspectrogram(path, plot=False):
    # Re-sample to SAMPLE_FREQ
    signal, sr = lb.load(path, sr=SAMPLE_FREQ, dtype=np.float32)
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*SAMPLE_FREQ)

    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*SAMPLE_FREQ) - n_sample,))))
    elif n_sample > n_sample_fit:
        signal = signal[(n_sample-n_sample_fit) // 2: (n_sample+n_sample_fit) // 2]
    melspect = lb.amplitude_to_db(lb.feature.melspectrogram(y=signal, sr=SAMPLE_FREQ, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2)

    if plot:
        melspect = melspect[np.newaxis, :]
        plt.imshow(melspect.reshape((melspect.shape[1], melspect.shape[2])))
        plt.show()

    return melspect


if __name__ == '__main__':
    log_scale_melspectrogram(sys.argv[1], True)
