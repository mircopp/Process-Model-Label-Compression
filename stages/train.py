from compression.sentence import SentenceCompressor

def train(X, y, use_synfeat=True):
    """
    Execute the training of the compression model and saves the model.
    :param X: The input data matrix
    :param y: The input labels
    :param use_synfeat: Use syntactical features?
    :return: None
    """
    model = SentenceCompressor()
    X_train, X_val, X_test, y_train, y_val, y_test = X[2000:], X[1000:2000], X[:1000], y[2000:], y[1000:2000], y[:1000]

    print('Start model training')
    model.compile(X.shape)
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate model performance
    metrics = model.calculate_metrics(X_test, y_test)
    print('Model precision: {}'.format(metrics['precision']))
    print('Model recall: {}'.format(metrics['recall']))
    print('Model F1 score: {}'.format(metrics['f1']))
    print('Model word accuracy: {}'.format(metrics['word_accuracy']))
    print('Model sentence accuracy: {}'.format(metrics['sentence_accuracy']))
    print('Model compression rate: {}'.format(metrics['compression_rate']))

    # Save the model
    model_name = 'sentence_compressor_3bilstm'
    if use_synfeat:
        model_name += '_synfeat'
    model.save_model(model_name)
