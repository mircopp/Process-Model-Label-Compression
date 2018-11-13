# Using Neural Networks to Derive Process Model Labels from Process Descriptions

## 1. Goal
> In this section the goal of the software in this repository is described. The code was implemented within the scope of my seminar work 'Using Neural Networks to Derive Process Model Labels from Process Descriptions' at the Institue for Business Informatics at DFKI.

In the area of business process modeling processes and work-flows are given as semi-structured models and/or textual descriptions. While some repositories of business processes make use of text descriptions others rely on models e.g. as the Event-Driven Process Chain (EPC). However, since text descriptions are mainly unstructured, the automated analysis can become complex. In this work, the author investigates an approach for deriving activity labels from textual process description to empower methods initially developed for the automated analysis of business process models to work on text-based process descriptions as well.
## 2. Documentation
### 2.1. SentencePreprocessor
The SentencePreprocessor is part of the compression.sentence module and was build to handle the whole preprocessing cyle of the raw data i.e. filtering, cleaning, lemmatizing, word embedding, POS and DEP encoding and sequence padding to transform the sequences of words into actual feature vectors. 
### 2.2. SentenceCompressor
The SentenceCompressor is part of the compression.sentence module as well and provides all interfaces that are necessary to create, compile, train and use a predictive model using keras. Additionally, this class provides functionality to calculate test statistics like accuracy, precision, recall, F1 score etc.
### 2.3. CompressionLanguageModel
The CompressionLanguageModel now is part of the module compression.language and gets initialized with a SentencePreprocessor object as well as a SentenceCompressor object. This class provides functioanlity to transforn a raw, unprocessed sequence of word tokens into a compressed sequence of word tokens using the predictive power of the SentenceCompressor.

## 3. Usage
### 3.1. Data
### 3.2. Tokenization
### 3.3. Preprocessing
### 3.4. Training
### 3.5. Application
### 3.6. Other
#### 3.6.1. Calculation of data set statistics
#### 3.6.2. Text similarity analysis


TODO Add description for tokenization, preprocessing, training and usage of the model here