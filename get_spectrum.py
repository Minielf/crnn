import collections
import os
import time
import sys
import numpy as np
from data_preprocess import *

def convert_to_spectrum(search_start_index = 180000, valid_index = 180000, segment_size=1000, id_filename = 'train_x_msd_id.txt', y_filename = 'train_y.npy'):
    ids = []
    with open(id_filename) as f:
        ids = f.read().split('\n')

    y_path = './' + y_filename
    y = np.load(y_path)

    spectrograms = []
    labels = []
    for index in range(search_start_index, len(ids)):
        if valid_index + segment_size > len(ids):
            return
        
        id = ids[index]
        audio_name = id + '.m4a'
        audio_path = './TRAIN_AUDIO_DATA/TRAIN_AUDIO_DATA_10/' + audio_name
        audio_exist = os.path.isfile(audio_path)
        
        if not audio_exist:
            continue

        try:
            spectrogram = log_scale_melspectrogram(audio_path)
            spectrograms.append(spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1], 1))
        except:
            continue

        labels.append(y[index])
        
        if len(spectrograms) % (segment_size//10) == 0:
            print("Next segment progress: ", len(spectrograms) / (segment_size))

        if len(spectrograms) == segment_size:
            new_valid_index = valid_index + segment_size
            spectrum_name = str(valid_index) + '_' + str(new_valid_index - 1) + '_spectrum.npy'
            spectrum_path = './TRAIN_SPECTRUM/TRAIN_SPECTRUM_10/' + spectrum_name

            labels_name = str(valid_index) + '_' + str(new_valid_index - 1) + '_label.npy'
            labels_path = './TRAIN_LABELS/TRAIN_LABEL_10/' + labels_name

            np.save(spectrum_path, np.stack(spectrograms).astype(np.float32))
            np.save(labels_path, np.stack(labels).astype(np.float32))
            print("Converted range (0 :", index ,") successfully at: " + spectrum_path)
            print("Next search_start_index =", index + 1)
            print("Next valid_index =", new_valid_index)

            spectrograms = []
            labels = []
            valid_index = new_valid_index

if __name__ == '__main__':
    convert_to_spectrum()
