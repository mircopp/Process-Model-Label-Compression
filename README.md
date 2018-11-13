# Using Neural Networks to Derive Process Model Labels from Process Descriptions

## 1. Goal
> In this section the goal of the software in this repository is described. The code was implemented within the scope of my seminar work 'Using Neural Networks to Derive Process Model Labels from Process Descriptions' at the Institue for Business Informatics at DFKI.

In the area of business process modeling processes and work-flows are given as semi-structured models and/or textual descriptions. While some repositories of business processes make use of text descriptions others rely on models e.g. as the Event-Driven Process Chain (EPC). However, since text descriptions are mainly unstructured, the automated analysis can become complex. In this work, the author investigates an approach for deriving activity labels from textual process description to empower methods initially developed for the automated analysis of business process models to work on text-based process descriptions as well.
## 2. Documentation
### 2.1. SentencePreprocessor
The SentencePreprocessor is part of the compression.sentence module and was build to handle the whole preprocessing cyle of the raw data i.e. filtering, cleaning, lemmatizing, word embedding, POS and DEP encoding and sequence padding to transform the sequences of words into actual feature vectors. 

This is an example how to initiate, fit and transform data with the preprocessor:
````python
from compression.sentence import SentencePreprocessor
import pickle as pc

preprocessor = SentencePreprocessor()

# Fit the preprocessor
preprocessor.fit(X)

# Transform into feature vectors
X, y = preprocessor.transform(X, y)

# Save the model as binary file
with open('model_binaries/sentence_preprocessor.bin', 'wb') as model_file:
    pc.dump(preprocessor, model_file)
    model_file.close()
````
### 2.2. SentenceCompressor
The SentenceCompressor is part of the compression.sentence module as well and provides all interfaces that are necessary to create, compile, train and use a predictive model using keras. Additionally, this class provides functionality to calculate test statistics like accuracy, precision, recall, F1 score etc.

Here is an example how to initiate, fit, and evaluate with the compression model:
````python
from compression.sentence import SentenceCompressor

model = SentenceCompressor()
X_train, X_val, X_test, y_train, y_val, y_test = X[2000:], X[1000:2000], X[:1000], y[2000:], y[1000:2000], y[:1000]
model.compile(X.shape)

# Train the model
model.fit(X_train, y_train, X_val, y_val)

# Evaluate model performance
metrics = model.calculate_metrics(X_test, y_test)
````
### 2.3. CompressionLanguageModel
The CompressionLanguageModel now is part of the module compression.language and gets initialized with a SentencePreprocessor object as well as a SentenceCompressor object. This class provides functioanlity to transforn a raw, unprocessed sequence of word tokens into a compressed sequence of word tokens using the predictive power of the SentenceCompressor.

This is how the language model can be used:
````python
import pickle as pc
from compression.sentence import SentenceCompressor
from compression.language import CompressionLanguageModel

use_synfeat = True
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

# use y as a matrix containing for each word the word, POS tag and DEP label
language_model.transform_sentence(y)
````
## 3. Usage
Please first install all the requirements given in [requirements.txt](requirements.txt) first.
```bash
pip3 install -r requirements.txt
```
### 3.1. Data
Before starting the tokenization process one should first make sure that all the data ressources are available in the ressources folder. The process descriptions are already integrated in the repository.

The google news corpus need to be downloaded [here](https://github.com/google-research-datasets/sentence-compression/tree/master/data), unzipped and moved into the folder **ressources/google**.
### 3.2. Tokenization
Before starting the [**main_tokenization.py**](main_tokenization.py) script one need to setup the docker container for syntax parsing using the [syntaxnet parser](https://github.com/tensorflow/models/tree/master/research/syntaxnet).
Therefore, one needs to build the docker container first by running the following commands:
```bash
docker build syntaxnet/ -t syntax_parser_rest
```
After this step the docker container needs to be started with this command:
```bash
 docker run -p 4000:80 sentence_parser_rest
```
Now one need to run the [main_tokenization.py](main_tokenization.py) file using this command and the sentences get parsed and stored as .csv files in the [csv folder](csv).
```bash
 python3 main_tokenization.py
```
### 3.3. Preprocessing
After tokenization all relevant files should be in the [csv folder](csv).
One can move on with the process by running the preprocessing file [main_preprocessing](main_preprocessing.py), which loads the parsed data, trains the SentencePreprocessor and finally transforms the raw data into the required format for input features for both models, the LK model and the base model.
After this step the SentencePreprocessor gets stored in the [model binaries folder](model_binaries)
```bash
python3 main_preprocessing.py
```
### 3.4. Training
After preprocessing the input features will be stored in 20 files in the [matrizes folder](matrizes).
These files will be loaded for training of our models during the execution of the [main_training.py](main_training.py) script. After this step the SentenceCompressor model gets stored in the [model binaries folder](model_binaries).
```bash
python3 main_training.py
```
### 3.5. Application
After all preprocessing and training is done one can make use of the pre-trained model by initializing the CompressionLanguageModel.

The script [main_usage.py](main_usage.py) loads all necessary parts and applies the model on the process description data.
```bash
python3 main_usage.py
```
### 3.6. Other
#### 3.6.1. Calculation of data set statistics
For the calculation of the relevant data set statistics please run the [dataset_statistics.py](dataset_statistics.py) file.
````bash
python3 dataset_statistics.py
````
#### 3.6.2. Text similarity analysis
For executing the text similarity analysis please run the file [text_similarity_analysis.py](text_similarity_analysis.py).
````bash
python3 text_similarity_analysis.py
````