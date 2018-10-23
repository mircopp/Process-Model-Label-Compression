import numpy as np

class CompressionLanguageModel():

    def __init__(self, preprocessor, compressor, syn_feat = False):
        # Initialize the language model to make predictions for new sentences
        self.preprocessor = preprocessor
        self.compressor = compressor
        self.syn_feat = syn_feat

    def get_compression(self, sentence, y):
        res = []
        for i in reversed(range(len(sentence))):
            if y[len(y) - 1 - i] == 1:
                res.append(sentence[len(sentence) - 1 - i])
        return np.array(res)

    def transform_sentence(self, y):
        # TODO Test functionality
        y_prep = self.preprocessor.transform(np.array([y]), pos=self.syn_feat, dep=self.syn_feat)[0]
        y_pred = self.compressor.predict(y_prep)
        return self.get_compression(y, y_pred.reshape(y_pred.shape[1]))
