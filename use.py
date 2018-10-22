import numpy as np
import pandas as pd
import pickle as pc
from compression.sentence import SentenceCompressor
from compression.language import CompressionLanguageModel

from preprocess import sentences_2D_to_3D, get_X_y

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
    with open('model_binaries/sentence_preprocessor.bin', 'rb') as model_file:
        print('Loading model')
        preprocessor = pc.load(model_file)
        model_file.close()

    compressor = SentenceCompressor()
    compressor.load_model()

    language_model = CompressionLanguageModel(preprocessor, compressor)
    for i in range(len(X)):
        current_sentence = X[i]
        print('Original sentence:', ' '.join(current_sentence[:, 0]))

        compressed_version = language_model.get_compression(current_sentence, y[i])
        print('Compressed sentence (Ground Truth):', ' '.join(compressed_version[:, 0]))

        predicted_version = language_model.transform_sentence(current_sentence)

        if len(predicted_version) > 0:
            predicted_version = np.array(predicted_version)
            print('Compressed sentence (predicted):', ' '.join(predicted_version[:, 0]))
        else:
            print('Compressed sentence (predicted): None')
        print('\n')

