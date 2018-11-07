from save import load_preprocessed_data
from stages.preprocess import sentences_2D_to_3D, get_X_y
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import pprint

def max_length (sequences):
    max = -1
    for x in sequences:
        max = len(x) if len(x) > max else max
    return max

def filter_maxlen(X, y=None, max_len = 100):
    idx_to_delete = []
    print('Number of sequences before filtering: {}'.format(len(X)))
    for i in range(len(X)):
        if len(X[i]) > max_len:
            idx_to_delete.append(i)
    if len(idx_to_delete) > 0:
        X = np.delete(X, idx_to_delete)
    if not(y is None):
        if len(idx_to_delete) > 0:
            y = np.delete(y, idx_to_delete)
    print('Number of sequences after filtering: {}'.format(len(X)))
    return X, y

def compute_stats_of_single_corpus(X, y, result):
    result['max_len_before'] = max_length(X)
    result['pairs_before'] = len(X)
    X, y = filter_maxlen(X, y)
    result['pairs'] = len(X)
    result['max_len'] = max_length(X)
    result['total_len'] = 0
    word_vocabulary = set()
    pos_vocabulary = set()
    dep_vocabulary = set()
    lemmatizer = WordNetLemmatizer()
    for i in range(len(X)):
        result['total_len'] += len(X[i])
        for j in range(len(X[i])):
            # replace numbers with numb token in the vocabulary
            if X[i][j][1] == 'CD':
                X[i][j][0] = '<NUMB>'
            else:
                X[i][j][0] = X[i][j][0]
            # Lemmatize the words to reduce vocabulary size
            try:
                X[i][j][0] = lemmatizer.lemmatize(X[i][j][0])
            except AttributeError:
                X[i][j][0] = X[i][j][0].decode('utf-8')
            except LookupError:
                import nltk
                nltk.download('wordnet')
                X[i][j][0] = lemmatizer.lemmatize(X[i][j][0])
            word_vocabulary.add(X[i][j][0])
            pos_vocabulary.add(X[i][j][1])
            dep_vocabulary.add(X[i][j][2])
    result['vocab'] = len(word_vocabulary)
    result['pos'] = len(pos_vocabulary)
    result['dep'] = len(dep_vocabulary)
    result['average_len'] = result['total_len'] / result['pairs']

    y_true = []
    for y_i in y:
        y_true.extend(y_i)
    y_true = np.array(y_true)
    condition = y_true == 1
    true_ratio = len(np.extract(condition, y_true)) / len(y_true)
    result['compression_rate'] = true_ratio
    return result, word_vocabulary, pos_vocabulary, dep_vocabulary, X, y

def compute_statistics(corpus, corpus_val):
    result = {}

    data_sentences = sentences_2D_to_3D(corpus)
    X, y = get_X_y(data_sentences)
    corpus_stats, word_vocabulary, pos_vocabulary, dep_vocabulary, X, y = compute_stats_of_single_corpus(X, y, {})
    result['corpus'] = corpus_stats

    data_sentences_val = sentences_2D_to_3D(corpus_val)
    X_val, y_val = get_X_y(data_sentences_val)
    corpus_val_stats, word_vocabulary_val, pos_vocabulary_val, dep_vocabulary_val, X_val, y_val = compute_stats_of_single_corpus(X_val, y_val, {})

    result['corpus_val'] = corpus_val_stats

    chunksize = 20
    total_set = []
    for i in range(chunksize):
        total_set.extend(X[i::chunksize].tolist())

    train_set = total_set[2000:]
    test_set = total_set[:1000]

    train_vocab = set()
    train_pos = set()
    train_dep = set()

    for seq in train_set:
        for word in seq:
            train_vocab.add(word[0])
            train_pos.add(word[1])
            train_dep.add(word[2])

    test_vocab = set()
    test_pos = set()
    test_dep = set()

    for seq in test_set:
        for word in seq:
            test_vocab.add(word[0])
            test_pos.add(word[1])
            test_dep.add(word[2])

    result['corpus']['train_vocab_coverage'] = len(train_vocab)/len(word_vocabulary)
    result['corpus']['train_pos_coverage'] = len(train_pos)/len(pos_vocabulary)
    result['corpus']['train_dep_coverage'] = len(train_dep)/len(dep_vocabulary)

    result['corpus']['train_test_vocab_coverage'] = len(test_vocab.intersection(train_vocab)) / len(test_vocab)
    result['corpus']['train_test_pos_coverage'] = len(test_pos.intersection(train_pos)) / len(test_pos)
    result['corpus']['train_test_dep_coverage'] = len(test_dep.intersection(train_dep)) / len(test_dep)

    result['corpus_val']['train_test_vocab_coverage'] = len(word_vocabulary_val.intersection(train_vocab)) / len(word_vocabulary_val)
    result['corpus_val']['train_test_pos_coverage'] = len(pos_vocabulary_val.intersection(train_pos)) / len(pos_vocabulary_val)
    result['corpus_val']['train_test_dep_coverage'] = len(dep_vocabulary_val.intersection(train_dep)) / len(dep_vocabulary_val)
    pprint.pprint(result)
    return result

if __name__ == '__main__':
    GOOGLE_DATA_SOURCES = ['sent-comp.train01.json.csv', 'sent-comp.train02.json.csv', 'sent-comp.train03.json.csv',
                           'sent-comp.train04.json.csv', 'sent-comp.train05.json.csv', 'sent-comp.train06.json.csv',
                           'sent-comp.train07.json.csv', 'sent-comp.train08.json.csv', 'sent-comp.train09.json.csv',
                           'sent-comp.train10.json.csv', 'comp-data.eval.json.csv']
    google_corpus = np.array([])
    for source in GOOGLE_DATA_SOURCES:
        tmp = pd.read_csv('csv/google/' + source, sep=',').values
        if google_corpus.any():
            google_corpus = np.concatenate((google_corpus, tmp), axis=0)
        else:
            google_corpus = tmp
    google_corpus = google_corpus[:, 1:]

    PD_DATA_SOURCES = ['process1.json.csv', 'process2.json.csv', 'process3.json.csv']
    pd_corpus = np.array([])
    for source in PD_DATA_SOURCES:
        tmp = pd.read_csv('csv/process_descriptions/' + source, sep=',').values
        if pd_corpus.any():
            pd_corpus = np.concatenate((pd_corpus, tmp), axis=0)
        else:
            pd_corpus = tmp
    pd_corpus = pd_corpus[:, 1:]

    stats = compute_statistics(google_corpus, pd_corpus)


