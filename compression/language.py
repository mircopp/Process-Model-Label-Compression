import numpy as np

class CompressionLanguageModel():

    def __init__(self, preprocessor, compressor, syn_feat = False):
        """
        Language model for generating compressions
        :param preprocessor: The pre-trained preprocessor.
        :param compressor: The pre-trained compression model.
        :param syn_feat: True if syn feat is used for the compression.
        """
        self.preprocessor = preprocessor
        self.compressor = compressor
        self.syn_feat = syn_feat

    def get_compression(self, sentence, y):
        """
        Get a compression for a sentence and given labels y
        :param sentence: The original sentence as an array.
        :param y: The labels of the sentences words
        :return: numpy.array with the compression
        """
        res = []
        for i in reversed(range(len(sentence))):
            if y[len(y) - 1 - i] == 1:
                res.append(sentence[len(sentence) - 1 - i])
        return np.array(res)

    def transform_sentence(self, y):
        """
        Transform a raw sentence y
        :param y: The raw sentence as a sequence of words with word, POS and DEP
        :return: The predicted compression.
        """
        y_prep = self.preprocessor.transform(np.array([y]), pos=self.syn_feat, dep=self.syn_feat)[0]
        y_pred = self.compressor.predict(y_prep)
        return self.get_compression(y, y_pred.reshape(y_pred.shape[1]))
