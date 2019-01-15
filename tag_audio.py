import sys
import os
import pickle as pk
import numpy as np
import sklearn.metrics as sm
import heapq
# import for get song
import collections
import requests
import time
import sys
from data_preprocess import *

import tensorflow as tf
import keras.backend as K
from keras import Sequential
from keras.layers import *

tags = ['rock', 'pop', 'alternative', 'indie', 'electronic',
        'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
        'beautiful', 'metal', 'chillout', 'male vocalists',
        'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
        '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
        'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
        'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
        '70s', 'party', 'country', 'easy listening',
        'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
        'Progressive rock', '60s', 'rnb', 'indie pop',
        'sad', 'House', 'happy']


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]

def CRNN():
    model = Sequential()
    model.add(ZeroPadding2D(padding=(0, 37), input_shape=(96, 1366, 1)))
    model.add(ZeroPadding2D(padding=(0, 37)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, strides=1, padding='same', name='conv1'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))
    model.add(Dropout(0.1, name='dropout1'))

    model.add(Conv2D(128, 3, strides=1, padding='same', name='conv2'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(3, 3), name='pool2'))
    model.add(Dropout(0.1, name='dropout2'))

    model.add(Conv2D(128, 3, strides=1, padding='same', name='conv3'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4), name='pool3'))
    model.add(Dropout(0.1, name='dropout3'))

    model.add(Conv2D(128, 3, strides=1, padding='same', name='conv4'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4), name='pool4'))
    model.add(Dropout(0.1, name='dropout4'))

    model.add(Reshape((15, 128)))
    model.add(GRU(32, return_sequences=True, name='gru1'))
    model.add(GRU(32, return_sequences=False, name='gru2'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='sigmoid', name='output'))
    return model

def get_spectrum(audio_path):
    audio_exist = os.path.isfile(audio_path)

    if not audio_exist:
        return None

    try:
        spectrogram = log_scale_melspectrogram(audio_path)
        spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1], 1)
        return spectrogram
    except:
        return None

def main_function():
    if len(sys.argv) >= 2:
        AUDIO_PATH = sys.argv[1]
    else:
        print("Need audio path!")
        return

    MODEL_PATH = "./trained_model.h5"
    model_exist = os.path.isfile(MODEL_PATH)
    model = CRNN()
    if model_exist:
        model.summary()
        model.load_weights(MODEL_PATH)
    else:
        print('Trained weights not found!')
        return

    # get the target file
    spectrum = get_spectrum(AUDIO_PATH) # 1 * 96 * 1366 * 1

    logits = model.predict(spectrum)
    heap = []
    for ind, score in enumerate(logits[0]):
        heapq.heappush(heap, (score, ind))
        if len(heap) > 10:
            heapq.heappop(heap)

    res = []
    for score, genre in heap:
        res += [tags[genre], score],
    res.sort(key = lambda x: x[1], reverse = True)
    print('Top ten genres are:\n', res)

if __name__ == '__main__':
    main_function()
