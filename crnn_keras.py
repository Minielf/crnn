import sys
import os
import pickle as pk
import numpy as np
import sklearn.metrics as sm

import tensorflow as tf
import keras.backend as K
from keras import Sequential
from keras.layers import *

batch_size = 10
n_samples = 1000
learning_rate = 0.003
n_epoch_per_file = 2
cv_split = 0.94
train_size = int(n_samples * cv_split)
valid_size = n_samples - train_size

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

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=[tf.keras.metrics.binary_accuracy])
    return model

if __name__ == '__main__':
    print(tf.VERSION)
    print(tf.keras.__version__)

    SPECTRUM_DIR = "./SPECTRUM/"
    LABELS_DIR = "./LABELS/"
    SAVE_PATH = "./trained_model.h5"

    if len(sys.argv) >= 2:
        SPECTRUM_DIR = sys.argv[1]
    if len(sys.argv) >= 3:
        LABELS_DIR = sys.argv[2]
    if len(sys.argv) >= 4:
        SAVE_PATH = sys.argv[3]
    
    for spectrum in os.listdir(SPECTRUM_DIR):
        # Reload model for clearing GPU memory
        model_exist = os.path.isfile(SAVE_PATH)
        model = CRNN()
        if model_exist:
            model.load_weights(SAVE_PATH)

        spectrum = os.path.splitext(spectrum)[0]
        seg = ""
        for i in spectrum:
            if i != 's':
                seg += i
            else:
                break
        label = seg + "label.npy"
        SPECTRUM_PATH = SPECTRUM_DIR + spectrum + ".npy"
        LABELS_PATH = LABELS_DIR + label

        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_indices = indices[0:train_size]
        valid_indices = indices[train_size:]

        spectrograms = np.load(SPECTRUM_PATH)
        labels = np.load(LABELS_PATH)
        X_train = spectrograms[train_indices]
        X_valid = spectrograms[valid_indices]
        y_train = labels[train_indices]
        y_valid = labels[valid_indices]

        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size).repeat()
        val_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        val_dataset = val_dataset.batch(batch_size).repeat()
        model.fit(dataset, 
                epochs=n_epoch_per_file, 
                steps_per_epoch= int(train_size / batch_size),
                validation_data=val_dataset,
                validation_steps= int(valid_size / batch_size))

        def calc_validate_auc(batch_size):
            indices = np.arange(len(X_valid))
            np.random.shuffle(indices)

            start = 0
            predictions = []
            while start < valid_size:
                input_indices = indices[start:min(start + batch_size, valid_size)]
                predictions.append(model.predict(X_valid[input_indices]))
                start += batch_size

            return sm.roc_auc_score(y_valid[indices], np.concatenate(predictions), average='samples')

        model.save_weights(SAVE_PATH)
        print('-------- Valid AUC : ', calc_validate_auc(20), '--------')

        K.clear_session()
        del model
