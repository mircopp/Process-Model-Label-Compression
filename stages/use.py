import pickle as pc
import numpy as np

from compression.sentence import SentenceCompressor
from compression.language import CompressionLanguageModel
from stages.preprocess import preprocess

def use(X, y, use_synfeat=True):
    """
    Apply the preprocessing and the compression model in X and check against ground truth y
    :param X: The parsed sentences.
    :param y: The true labels.
    :param use_synfeat: Use syntactical features?
    :return: None
    """
    with open('model_binaries/sentence_preprocessor.bin', 'rb') as model_file:
        print('Loading model')
        preprocessor = pc.load(model_file)
        model_file.close()

    compressor = SentenceCompressor()
    if use_synfeat:
        compressor.load_model(model_name='sentence_compressor_3bilstm_synfeat')
    else:
        compressor.load_model(model_name='sentence_compressor_3bilstm')
    compressor.model.summary()
    X_pd, y_pd = preprocessor.transform(X, y, use_synfeat, use_synfeat)
    metrics = compressor.calculate_metrics(X_pd, y_pd)
    print('Model precision: {}'.format(metrics['precision']))
    print('Model recall: {}'.format(metrics['recall']))
    print('Model F1 score: {}'.format(metrics['f1']))
    print('Model word accuracy: {}'.format(metrics['word_accuracy']))
    print('Model sentence accuracy: {}'.format(metrics['sentence_accuracy']))
    print('Model compression rate: {}'.format(metrics['compression_rate']))


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


def get_compressions(X, use_synfeat=False):
    """
    Get the compression for X
    :param X: Input feature matrix
    :param use_synfeat: Use syntactical features?
    :return: The compressed sequences of words
    """
    with open('model_binaries/sentence_preprocessor.bin', 'rb') as model_file:
        print('Loading model')
        preprocessor = pc.load(model_file)
        model_file.close()

    compressor = SentenceCompressor()
    if use_synfeat:
        compressor.load_model(model_name='sentence_compressor_3bilstm_synfeat')
    else:
        compressor.load_model(model_name='sentence_compressor_3bilstm')
    compressor.model.summary()

    language_model = CompressionLanguageModel(preprocessor, compressor, syn_feat=use_synfeat)

    result = []
    for i in range(len(X)):
        current_sentence = X[i]
        predicted_version = language_model.transform_sentence(current_sentence)
        if len(predicted_version) > 0:
            predicted_version = np.array(predicted_version)
            print('Compressed sentence (predicted, synfeat=' + str(use_synfeat) + '):', ' '.join(predicted_version[:, 0]))
        else:
            print('Compressed sentence (predicted, synfeat=' + str(use_synfeat) + '): None')
        print('\n')
        result.append(predicted_version)
    return result

