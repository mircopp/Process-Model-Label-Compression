import numpy as np
import pandas as pd
import pickle as pc
from stages.use import use
from compression.sentence import SentenceCompressor
from compression.language import CompressionLanguageModel

from stages.preprocess import sentences_2D_to_3D, get_X_y

if __name__ == '__main__':
    print('Loading process data')
    PD_DATA_SOURCES = ['process1.json.csv', 'process2.json.csv', 'process3.json.csv']
    pd_corpus = np.array([])
    for source in PD_DATA_SOURCES:
        tmp = pd.read_csv('csv/process_descriptions/' + source, sep=',').values
        if pd_corpus.any():
            pd_corpus = np.concatenate((pd_corpus, tmp), axis=0)
        else:
            pd_corpus = tmp
    pd_corpus = pd_corpus[:, 1:]

    data_sentences = sentences_2D_to_3D(pd_corpus)
    X, y = get_X_y(data_sentences)

    use(X, y, use_synfeat=False)
    use(X, y, use_synfeat=True)
