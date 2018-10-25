import pickle as pc
import numpy as np

from compression.sentence import SentenceCompressor
from compression.language import CompressionLanguageModel

def use(X, y, use_synfeat=False):
    with open('model_binaries/sentence_preprocessor.bin', 'rb') as model_file:
        print('Loading model')
        preprocessor = pc.load(model_file)
        model_file.close()

    compressor = SentenceCompressor()
    if use_synfeat:
        compressor.load_model(model_name='sentence_compressor_3bilstm_synfeat')
    else:
        compressor.load_model(model_name='sentence_compressor_3bilstm')

    language_model = CompressionLanguageModel(preprocessor, compressor, syn_feat=use_synfeat)

    for i in range(len(X)):
        current_sentence = X[i]
        print('Original sentence:', ' '.join(current_sentence[:, 0]))

        compressed_version = language_model.get_compression(current_sentence, y[i])
        print('Compressed sentence (Ground Truth):', ' '.join(compressed_version[:, 0]))

        predicted_version = language_model.transform_sentence(current_sentence)
        if len(predicted_version) > 0:
            predicted_version = np.array(predicted_version)
            predicted_version = np.array(predicted_version)
            print('Compressed sentence (predicted, synfeat=' + str(use_synfeat) + '):', ' '.join(predicted_version[:, 0]))
        else:
            print('Compressed sentence (predicted, synfeat=' + str(use_synfeat) + '): None')
        print('\n')

