import numpy as np
import pandas as pd
import pickle as pc
import json
from compression.sentence import SentencePreprocessor
from save import save_preprocessed_data

def sentences_2D_to_3D (X):
    res = []
    seq = []
    for x in X:
        seq.append(x)
        if x[0] == '<EOS>':
            res.append(seq)
            seq = []
    return np.array(res)

def get_X_y (data):
    X, y = [], []
    for x in data:
        curr_X = []
        curr_y = []
        for x_i in x:
            curr_X.append(np.array([x_i[0], x_i[1], x_i[2]]))
            curr_y.append(np.array(x_i[-1]))
        X.append(np.array(curr_X))
        y.append(np.array(curr_y))
    return np.array(X), np.array(y)

if __name__ == '__main__':
    print('Loading data')
    CROSS_VAL_FACTOR = 1

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
    corpus = np.concatenate((google_corpus, pd_corpus))

    data_sentences = sentences_2D_to_3D(corpus)
    X, y = get_X_y(data_sentences)
    try:
        with open('model_binaries/sentence_preprocessor.bin', 'rb') as model_file:
            print('Loading model')
            preprocessor = pc.load(model_file)
            model_file.close()
    except FileNotFoundError:
        print('Fitting model')
        preprocessor = SentencePreprocessor()
        preprocessor.fit(X)
        with open('model_binaries/sentence_preprocessor.bin', 'wb') as model_file:
            pc.dump(preprocessor, model_file)
            model_file.close()

    print('Tansforming data')
    # Start with google data
    google_sentences = sentences_2D_to_3D(google_corpus)
    X_google, y_google = get_X_y(google_sentences)

    X_google, y_google = preprocessor.transform(X_google[::CROSS_VAL_FACTOR], y_google[::CROSS_VAL_FACTOR])
    y_google = y_google.reshape(y_google.shape[0], y_google.shape[1], 1)

    # Preprocess process description data
    pd_sentences = sentences_2D_to_3D(pd_corpus)
    X_pd, y_pd = get_X_y(pd_sentences)

    X_pd, y_pd = preprocessor.transform(X_pd, y_pd)
    y_pd = y_pd.reshape(y_pd.shape[0], y_pd.shape[1], 1)

    print('Save preprocessed data in chunks')
    save_preprocessed_data(X_google, name='preprocessed_X.google')
    save_preprocessed_data(y_google, name='preprocessed_y.google')
    save_preprocessed_data(X_pd, chunksize=1, name='preprocessed_X.pd')
    save_preprocessed_data(y_pd, chunksize=1, name='preprocessed_y.pd')

    print('Finished preprocessing')
