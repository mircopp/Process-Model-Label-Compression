from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import multiprocessing

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Dropout, Bidirectional
from keras.optimizers import Adam
from keras import backend as K
from keras.models import model_from_json

import matplotlib.pyplot as plt


class SentencePreprocessor:

    def __init__(self, lemmatize=True, numb_tokens=True, embedding_size=200, max_len = 100):
        """
        Initializes a SentencePreprocessor object
        :param lemmatize: Use lammatizing?
        :param numb_tokens: Clean data with NUMB tokens?
        :param embedding_size: Dimensionality of word embeddings
        :param max_len: Length of padded sequences
        """
        self.lemmatize = lemmatize
        self.numb_tokens = numb_tokens
        self.embedding_size = embedding_size
        self.max_len = max_len

        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

        self.word_vocabulary = set()
        self.pos_vocabulary = set()
        self.dep_vocabulary = set()

    def fit (self, X):
        """
        Fits the preprocessor to the given vocabulary in X
        :param X: The tokenized sequences of words
        :return: None
        """
        longest_sequence = max_length(X)
        self.max_len = longest_sequence if longest_sequence < self.max_len else self.max_len
        for x in X:
            for x_i in x:
                if self.numb_tokens:
                    # replace numbers with numb token in the vocabulary
                    if x_i[1] == 'CD':
                        current_word = '<NUMB>'
                    else:
                        current_word = x_i[0]
                if self.lemmatize:
                    try:
                        current_word = self.lemmatizer.lemmatize(current_word)
                    except AttributeError:
                        current_word = current_word.decode('utf-8')
                    except LookupError:
                        import nltk
                        nltk.download('wordnet')
                        current_word = self.lemmatizer.lemmatize(current_word)
                self.word_vocabulary.add(current_word)
                self.pos_vocabulary.add(x_i[1])
                self.dep_vocabulary.add(x_i[2])

        # Train word2vec model with standard params from gated neural networks approach
        train_data = [[word] for word in self.word_vocabulary]
        emb_dim = self.embedding_size
        n_exposures = 0
        window_size = 7
        cpu_count = multiprocessing.cpu_count()

        # build word2vec(skip-gram) model for X word corpus
        self.word2vec = Word2Vec(size=emb_dim,
                         min_count=n_exposures,
                         window=window_size,
                         workers=cpu_count,
                         iter=10,
                         sg=1)
        self.word2vec.build_vocab(train_data)
        self.word2vec.train(train_data, total_examples=len(train_data), epochs=5)

        # Train one hot encoding for pos labels
        self.pos2vec = LabelBinarizer()
        self.pos2vec.fit([pos for pos in self.pos_vocabulary])

        # Train one hot encoding for dependency labels
        self.dep2vec = LabelBinarizer()
        self.dep2vec.fit([dep for dep in self.dep_vocabulary])

    def transform(self, X, y = None, pos=True, dep=True):
        """
        Transforms the matrix X and label vector y into feature vectors
        :param X: Tokenized sequences of words
        :param y: Labels for sequences
        :param pos: Use POS tags?
        :param dep: Use DEP labels?
        :return: X, y as feature vectors
        """
        # First step: delete all sequences that are longer than to expect (these much likely are wrongly parsed)
        X, y = self._filter_maxlen(X, y)
        longest_sequence = max_length(X)
        print('Longest sequence is {} words.'.format(longest_sequence))

        embedding_dimension = self.embedding_size
        ranges = [0, self.embedding_size]
        if pos:
            embedding_dimension += len(self.pos2vec.classes_)
            ranges.append(ranges[len(ranges)-1] + len(self.pos2vec.classes_))
        if dep:
            embedding_dimension += len(self.dep2vec.classes_)
            ranges.append(ranges[len(ranges)-1] + len(self.dep2vec.classes_))

        X_sequences = np.zeros((len(X), self.max_len, embedding_dimension))
        if not(y is None):
            y_sequence = np.zeros((len(y), self.max_len))
        else:
            y_sequence = None

        for i in range(len(X)):
            for j in range(len(X[i])):
                if self.numb_tokens:
                    # replace numbers with numb token in the vocabulary
                    if X[i][j][1] == 'CD':
                        current_word = '<NUMB>'
                    else:
                        current_word = X[i][j][0]
                if self.lemmatize:
                    # Lemmatize the words to reduce vocabulary size
                    try:
                        current_word = self.lemmatizer.lemmatize(current_word)
                    except AttributeError:
                        current_word = current_word.decode('utf-8')
                    except LookupError:
                        import nltk
                        nltk.download('wordnet')
                        current_word = self.lemmatizer.lemmatize(current_word)
                sequence_position = X_sequences.shape[1] - (len(X[i]) - j)
                X_sequences[i, sequence_position, ranges[0]:ranges[1]] = self._vectorize_word(current_word)
                if pos:
                    X_sequences[i, sequence_position, ranges[1]:ranges[2]] = self._vectorize_pos_label(X[i][j][1])
                    if dep:
                        X_sequences[i, sequence_position, ranges[2]:ranges[3]] = self._vectorize_dep_label(X[i][j][2])
                else:
                    if dep:
                        X_sequences[i, sequence_position, ranges[1]:ranges[2]] = self._vectorize_dep_label(X[i][j][2])
            if not(y is None):
                y_sequence[i, (len(y_sequence[i])-len(y[i])):] = y[i]

        return X_sequences, y_sequence

    def fit_transform(self, X, y, pos=True, dep=True):
        """
        Executes first fit then transform.
        :param X: Tokenized sequences of words
        :param y: Labels for sequences
        :param pos: Use POS tags?
        :param dep: Use DEP labels?
        :return: X, y as feature vectors
        """
        self.fit(X)
        return self.transform(X, y, pos=pos, dep=dep)

    def _filter_maxlen(self, X, y=None):
        idx_to_delete = []
        print('Number of sequences before filtering: {}'.format(len(X)))
        for i in range(len(X)):
            if len(X[i]) > self.max_len:
                idx_to_delete.append(i)
        if len(idx_to_delete) > 0:
            X = np.delete(X, idx_to_delete)
        if not(y is None):
            if len(idx_to_delete) > 0:
                y = np.delete(y, idx_to_delete)
        print('Number of sequences after filtering: {}'.format(len(X)))
        return X, y


    def _vectorize_words(self, X_word):
        for i in range(len(X_word)):
            for j in range(len(X_word[i])):
                X_word[i][j] = self.word2vec.wv[X_word[i][j]]
        return X_word

    def _vectorize_word(self, word):
        return self.word2vec.wv[word]
    
    def _vectorize_pos(self, X_pos):
        for i in range(len(X_pos)):
            X_pos[i] = self.pos2vec.transform(X_pos[i])
        return X_pos

    def _vectorize_pos_label(self, pos_label):
        return self.pos2vec.transform([pos_label])[0]

    def _vectorize_dep(self, X_dep):
        for i in range(len(X_dep)):
            X_dep[i] = self.dep2vec.transform(X_dep[i])
        return X_dep

    def _vectorize_dep_label(self, dep_label):
        return self.dep2vec.transform([dep_label])

def max_length (sequences):
    max = -1
    for x in sequences:
        max = len(x) if len(x) > max else max
    return max

class SentenceCompressor:

    def __init__(self):
        """
        Initializes the SentenceCompressor model.
        """
        self.model = Sequential()
        self.batch_size = 32

    def compile(self, input_shape):
        """
        Compiles the model.
        :param input_shape: The shape of the input data.
        :return: None
        """
        self.shape = input_shape

        # Add Bi-LSTM Layer
        self.model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.shape[1], self.shape[2])))
        # Add dropout layer in between to avoid overfitting
        self.model.add(Dropout(0.5))

        # Add Bi-LSTM Layer
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        # Add dropout layer in between to avoid overfitting
        self.model.add(Dropout(0.5))

        # Add Bi-LSTM Layer
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        # Add dropout layer in between to avoid overfitting
        self.model.add(Dropout(0.5))

        self.model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        self.model.compile(optimizer=Adam(lr=0.001), metrics=['accuracy'], loss=nll1)

        self.model.summary()

    def fit(self, X_train, y_train, X_val, y_val, n_epochs = 10, plot_history = True):
        """
        Execute the model training.
        :param X_train: Training feature matrix
        :param y_train: Training labels
        :param X_val: Validation feature matrix
        :param y_val: Validation labels
        :param n_epochs: Number of epochs
        :param plot_history: Plot the training history?
        :return: None
        """
        self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=n_epochs, validation_data=(X_val, y_val))
        if plot_history:
            hist = pd.DataFrame(self.history.history)
            print(hist)
            plt.style.use("ggplot")
            plt.figure(figsize=(7, 7))
            ax = plt.subplot(111)
            ax.plot(hist["acc"], label='accuracy')
            ax.plot(hist["val_acc"], label='validation accuracy')
            ax.legend()
            plt.xlabel('Number of epochs')
            plt.ylabel('accuracy score')
            plt.title('Training history')
            plot_name = '3bilstm_training_history'
            if self.shape[2] > 200:
                plot_name += '_synfeat'

            plt.savefig(plot_name + '.png')

    def predict(self, X):
        """
        Predict on features X.
        :param X: The feature matrix to predict on
        :return: The predicted deletion labels.
        """
        return self.model.predict_classes(X, batch_size=self.batch_size)

    def evaluate(self, X_test, y_test):
        """
        Autmoatic evaluation calculating the loss function and accuracy.
        :param X_test: Test feature matrix
        :param y_test: Test labels
        :return: The loss function score and the accuracy score
        """
        # Automatic evaluation with default accuracy metric
        score, acc = self.model.evaluate(X_test, y_test,
                                         batch_size=self.batch_size)
        return score, acc

    def calculate_metrics(self, X_test, y_test):
        """
        Custom evaluation computing the relevant metrics.
        :param X_test: Test feature matrix
        :param y_test: Test labels
        :return: The metric scores as a dict.
        """
        return {
            'precision' : self.precision_score(X_test, y_test),
            'recall' : self.recall_score(X_test, y_test),
            'f1' : self.f1_score(X_test, y_test),
            'word_accuracy' : self.word_accuracy(X_test, y_test),
            'sentence_accuracy' : self.per_sentence_accuracy(X_test, y_test),
            'compression_rate' : self.compression_rate(X_test, y_test),
        }

    def save_model(self, model_name = 'sentence_compressor'):
        """
        Saves the models weights and architecture
        :param model_name: The name of the model to be saved.
        :return:
        """
        model_json = self.model.to_json()
        with open('model_binaries/' + model_name + '.json', 'w') as json_file:
            json_file.write(model_json)
            json_file.close()
        self.model.save_weights('model_binaries/' + model_name + '.h5')
        print('Saved keras model')

    def load_model(self, model_name = 'sentence_compressor'):
        """
        Loads a pretrained model from weights and architecture json and re-compiles it.
        :param model_name: The name of the pre-trained model
        :return: None
        """
        # load json and create model
        json_file = open('model_binaries/' + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights('model_binaries/' + model_name + '.h5')
        self.model.compile(loss=nll1, optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])
        print("Loaded model from disk")

    def f1_score(self, X_test, y_test):
        """
        Compute f1 score.
        :param X_test: Test feature matrix
        :param y_test: Test labels
        :return: F1 score
        """
        y_pred, y_true = self._y_pred_y_true(X_test, y_test)
        return f1_score(y_true, y_pred)

    def precision_score(self, X_test, y_test):
        """
        Compute precision score.
        :param X_test: Test feature matrix
        :param y_test: Test labels
        :return: Precision score
        """
        y_pred, y_true = self._y_pred_y_true(X_test, y_test)
        return precision_score(y_true, y_pred)

    def recall_score(self, X_test, y_test):
        """
        Compute recall score.
        :param X_test: Test feature matrix
        :param y_test: Test labels
        :return: Recall score
        """
        y_pred, y_true = self._y_pred_y_true(X_test, y_test)
        return recall_score(y_true, y_pred)

    def per_sentence_accuracy(self, X_test, y_test):
        """
        Compute sentence-based accuracy score.
        :param X_test: Test feature matrix
        :param y_test: Test labels
        :return: Sentence-based accuracy score
        """
        y_pred = self.predict(X_test)
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
        pred_ref = []
        true_ref = []
        for i in range(len(X_test)):
            start = self._find_start_point(X_test[i])
            pred_ref.append(y_pred[i, start:])
            true_ref.append(y_test[i, start:])
        correctly_classified = 0
        for j in range(len(pred_ref)):
            if np.array_equal(pred_ref[j], true_ref[j]):
                correctly_classified += 1
        return correctly_classified/len(y_pred)

    def word_accuracy(self, X_test, y_test):
        """
        Compute word-based accuracy score.
        :param X_test: Test feature matrix
        :param y_test: Test labels
        :return: Word-based accuracy score
        """
        y_pred, y_true = self._y_pred_y_true(X_test, y_test)
        return accuracy_score(y_true, y_pred)

    def compression_rate(self, X_test, y_test):
        """
        Compute compression rate.
        :param X_test: Test feature matrix
        :param y_test: Test labels
        :return: Compression rate for predicted values as well as for the real labels given as y_test
        """
        y_pred, y_true = self._y_pred_y_true(X_test, y_test)
        pred_condition = y_pred == 1
        true_condition = y_true == 1
        pred_ratio = len(np.extract(pred_condition, y_pred)) / len(y_pred)
        true_ratio = len(np.extract(true_condition, y_true)) / len(y_true)
        return {'true' : true_ratio, 'predicted': pred_ratio}

    def _y_pred_y_true(self, X, y):
        y_pred = self.predict(X)
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
        y_test = y.reshape(y.shape[0], y.shape[1])
        pred_ref = np.array([])
        true_ref = np.array([])
        for i in range(len(X)):
            start = self._find_start_point(X[i])
            pred_ref = np.concatenate((pred_ref, y_pred[i, start:]))
            true_ref = np.concatenate((true_ref, y_test[i, start:]))
        return pred_ref, true_ref


    def _find_start_point(self, x):
        start_idx = -1
        for i in range(len(x)):
            if not self._is_zero_vec(x[i]):
                return i
            else:
                continue
        return None

    def _is_zero_vec(self, x):
        for x_i in x:
            if x_i != 0:
                return False
            else:
                continue
        return True

def nll1(y_true, y_pred):
    """ Negative log likelihood. """

    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)