import sys
import os
import pickle as pk
import numpy as np
import tensorflow as tf
import sklearn.metrics as sm

batch_size = 50
learning_rate = 0.003
n_epoch = 50
n_samples = 1000  # change to 1000 for entire dataset
cv_split = 0.8
train_size = int(n_samples * cv_split)
valid_size = n_samples - train_size
layers = tf.layers

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


class CRNN:
    def __init__(self):
        self.input = tf.placeholder("float", [None, 96, 1366, 1])
        self.labels = tf.placeholder("float", [None, len(tags)])
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        self.logits = self.forward_pass()
        self.loss = self.loss_function()
        self.train = self.back_propagation()

    def forward_pass(self):
        keep_rate_cnn = tf.where(self.phase_train, tf.constant(1.0), tf.constant(0.1))

        x = tf.cast(tf.pad(self.input, [[0, 0], [0, 0], [37, 37], [0, 0]], 'CONSTANT'), tf.float32)
        x = layers.batch_normalization(x, training=self.phase_train)

        conv2_1 = layers.conv2d(x, 64, 3, strides=1, padding='SAME')
        conv2_1 = tf.nn.relu(layers.batch_normalization(conv2_1, training=self.phase_train))
        mpool_1 = layers.max_pooling2d(conv2_1, pool_size=2, strides=2, padding='VALID')
        dropout_1 = layers.dropout(mpool_1, keep_rate_cnn)

        conv2_2 = layers.conv2d(dropout_1, 128, 3, strides=1, padding='SAME')
        conv2_2 = tf.nn.relu(layers.batch_normalization(conv2_2, training=self.phase_train))
        mpool_2 = layers.max_pooling2d(conv2_2, pool_size=3, strides=3, padding='VALID')
        dropout_2 = layers.dropout(mpool_2, keep_rate_cnn)

        conv2_3 = layers.conv2d(dropout_2, 128, 3, strides=1, padding='SAME')
        conv2_3 = tf.nn.relu(layers.batch_normalization(conv2_3, training=self.phase_train))
        mpool_3 = layers.max_pooling2d(conv2_3, pool_size=4, strides=4, padding='VALID')
        dropout_3 = layers.dropout(mpool_3, keep_rate_cnn)

        conv2_4 = layers.conv2d(dropout_3, 128, 3, strides=1, padding='SAME')
        conv2_4 = tf.nn.relu(layers.batch_normalization(conv2_4, training=self.phase_train))
        mpool_4 = layers.max_pooling2d(conv2_4, pool_size=4, strides=4, padding='VALID')
        dropout_4 = layers.dropout(mpool_4, keep_rate_cnn)

        FINAL_CNN_SIZE = [1, 15, 128]
        RNN_SZ = 32
        gru1_in = tf.reshape(dropout_4, [-1, FINAL_CNN_SIZE[1], FINAL_CNN_SIZE[2]])
        gru1 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(RNN_SZ) for _ in range(15)])
        gru1_out, state = tf.nn.dynamic_rnn(gru1, gru1_in, dtype=tf.float32, scope='gru1')

        gru2 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(RNN_SZ) for _ in range(15)])
        gru2_out, state = tf.nn.dynamic_rnn(gru2, gru1_out, dtype=tf.float32, scope='gru2')
        gru2_out = tf.gather(gru2_out, FINAL_CNN_SIZE[1] - 1, axis=1)

        keep_rate_rnn = tf.where(self.phase_train, tf.constant(1.0), tf.constant(0.3))
        dropout_5 = layers.dropout(gru2_out, keep_rate_rnn)

        logits = tf.layers.dense(dropout_5, len(tags), activation=tf.nn.sigmoid)
        return logits

    def loss_function(self):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

    def back_propagation(self):
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        return train_op


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]


if __name__ == '__main__':
    for spectrum in os.listdir("./SPECTRUM"):
        spectrum = os.path.splitext(spectrum)[0]
        seg = ""
        for i in spectrum:
            if i != 's':
                seg += i
            else:
                break
        label = seg + "label.npy"
        SPECTRUM_PATH = "./SPECTRUM/" + spectrum + ".npy"
        LABELS_PATH = "./LABELS/" + label
    # SPECTRUM_PATH = "./SPECTRUM/0_999_spectrum.npy"
    # LABELS_PATH = "./LABELS/0_999_label.npy"
    # if len(sys.argv) >= 2:
    #     SPECTRUM_PATH = sys.argv[1]
    # if len(sys.argv) >= 3:
    #     LABELS_PATH = sys.argv[2]

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

        model = CRNN()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        def calc_validate_auc(batch_size):
            indices = np.arange(len(X_valid))
            np.random.shuffle(indices)

            start = 0
            predictions = []
            while start < valid_size:
                input_indices = indices[start:min(start + batch_size, valid_size)]
                valid_input_dict = {model.input: X_valid[input_indices],
                                    model.labels: y_valid[input_indices],
                                    model.phase_train: False}
                prediction = sess.run(model.logits, feed_dict=valid_input_dict)
                predictions.append(prediction)
                start += batch_size

            return sm.roc_auc_score(y_valid[indices], np.concatenate(predictions), average='samples')


        for i in range(n_epoch):
            training_batch = zip(range(0, train_size, batch_size), range(batch_size, train_size + 1, batch_size))
            j = 0
            batches_per_output = 4
            for start, end in training_batch:
                train_input_dict = {model.input: X_train[start:end],
                                    model.labels: y_train[start:end],
                                    model.phase_train: True}
                _, loss = sess.run([model.train, model.loss], feed_dict=train_input_dict)

                if j % (batches_per_output) == 0:
                    print("Loss for batch", j, ":", loss)
                j += 1

            print('------- Epoch :', i, 'AUC : ', calc_validate_auc(20), '--------')
            # print sort_result(tags, predictions)[:5])
        saver.save(sess, './project_saved_model')
