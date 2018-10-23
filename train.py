import numpy as np
import pandas as pd
import pickle as pc
from save import load_preprocessed_data
from compression.sentence import SentenceCompressor

from preprocess import sentences_2D_to_3D, get_X_y

if __name__ == '__main__':
    try:
        X = load_preprocessed_data(name='preprocessed_X.google')
        y = load_preprocessed_data(name='preprocessed_y.google')
    except FileNotFoundError:
        print('No stored data found, computing from raw.')
        GOOGLE_DATA_SOURCES = ['sent-comp.train01.json.csv', 'sent-comp.train02.json.csv', 'sent-comp.train03.json.csv',
                               'sent-comp.train04.json.csv', 'sent-comp.train05.json.csv', 'sent-comp.train06.json.csv',
                               'sent-comp.train07.json.csv', 'sent-comp.train08.json.csv', 'sent-comp.train09.json.csv',
                               'sent-comp.train10.json.csv', 'comp-data.eval.json.csv']
        CROSS_VAL_FACTOR = 25
        data = np.array([])
        for source in GOOGLE_DATA_SOURCES:
            tmp = pd.read_csv('csv/google/' + source, sep=',').values
            if data.any():
                data = np.concatenate((data, tmp), axis=0)
            else:
                data = tmp
        data = data[:, 1:]
        data_sentences = sentences_2D_to_3D(data)
        del data
        X, y = get_X_y(data_sentences)
        with open('model_binaries/sentence_preprocessor.bin', 'rb') as model_file:
            print('Loading model')
            preprocessor = pc.load(model_file)
            model_file.close()

        print('Tansforming data')
        # get a chunk of the data
        X, y = preprocessor.transform(X[::CROSS_VAL_FACTOR], y[::CROSS_VAL_FACTOR])

        # Train sequence to sequence rnn model
        # Split into train and test set
        y = y.reshape(y.shape[0], y.shape[1], 1)

    X_train, X_val, X_test, y_train, y_val, y_test = X[2000:], X[1000:2000], X[:1000], y[2000:], y[1000:2000], y[:1000]

    print('Model training')
    model = SentenceCompressor()
    model.compile(X.shape, crf=True)
    model.fit(X_train, y_train, X_val, y_val)
    score, acc = model.evaluate(X_test, y_test)
    print('Model score: {}'.format(score))
    print('Model accuracy: {}'.format(acc))

    metrics = model.calculate_metrics(X_test, y_test)
    print('Model precision: {}'.format(metrics['precision']))
    print('Model recall: {}'.format(metrics['recall']))
    print('Model F1 score: {}'.format(metrics['f1']))
    print('Model word accuracy: {}'.format(metrics['word_accuracy']))
    print('Model sentence accuracy: {}'.format(metrics['sentence_accuracy']))
    print('Model compression rate: {}'.format(metrics['compression_rate']))

    # Save the model
    model.save_model()

    # # Load the compressor from disk
    # loaded_model = SentenceCompressor()
    # loaded_model.load_model()
    #
    # # Manually test the compressor from disk
    # test_index = randint(0, len(X))
    #
    # data_sents = data_sentences[::CROSS_VAL_FACTOR]
    # test_index = randint(0, len(data_sents))
    # test_sentence = np.array(data_sents[test_index])
    # print('Original sentence:', ' '.join(test_sentence[:, 0]))
    #
    # compressed_version = get_compression(test_sentence, y[test_index])
    # print('\nCompressed sentence (Ground Truth):', ' '.join(compressed_version[:, 0]))
    #
    # y_pred = loaded_model.predict(X[test_index].reshape(1, X.shape[1], X.shape[2]))
    # predicted_version = get_compression(test_sentence, y_pred.reshape(y_pred.shape[1]))
    #
    # if len(predicted_version) > 0:
    #     predicted_version = np.array(predicted_version)
    #     print('\nCompressed sentence (predicted):', ' '.join(predicted_version[:, 0]))
    # else:
    #     print('\nCompressed sentence (predicted): None')
