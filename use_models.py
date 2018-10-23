import numpy as np
import pandas as pd
import pickle as pc
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
    with open('model_binaries/sentence_preprocessor_synfeat.bin', 'rb') as model_file:
        print('Loading model')
        preprocessor = pc.load(model_file)
        model_file.close()

    syn_compressor = SentenceCompressor()
    syn_compressor.load_model(model_name='sentence_compressor_3bilstm_synfeat')

    emb_compressor = SentenceCompressor()
    emb_compressor.load_model(model_name='sentence_compressor_3bilstm')

    syn_language_model = CompressionLanguageModel(preprocessor, syn_compressor,syn_feat=True)
    emb_language_model = CompressionLanguageModel(preprocessor, emb_compressor, syn_feat=False)
    for i in range(len(X)):
        current_sentence = X[i]
        print('Original sentence:', ' '.join(current_sentence[:, 0]))

        compressed_version = syn_language_model.get_compression(current_sentence, y[i])
        print('Compressed sentence (Ground Truth):', ' '.join(compressed_version[:, 0]))

        predicted_version_syn = syn_language_model.transform_sentence(current_sentence)
        if len(predicted_version_syn) > 0:
            predicted_version_syn = np.array(predicted_version_syn)
            print('Compressed sentence (predicted, syntactic):', ' '.join(predicted_version_syn[:, 0]))
        else:
            print('Compressed sentence (predicted, syntactic): None')

        predicted_version_emb = emb_language_model.transform_sentence(current_sentence)
        if len(predicted_version_emb) > 0:
            predicted_version_emb = np.array(predicted_version_emb)
            print('Compressed sentence (predicted, embedding):', ' '.join(predicted_version_emb[:, 0]))
        else:
            print('Compressed sentence (predicted, embedding): None')
        print('\n')

