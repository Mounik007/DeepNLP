# Deep NLP

Deep NLP is a dataset that was released on Kaggle. The dataset consists of two different tasks, one to predict if there will be a response from a chatbot given the text or if the user will be redirected to help. The second task is to see if the resume has been flagged or unflagged. If the particular resume has been flagged, then the candidate is selected for the interview, if not the candidate can modify his resume.

Currently, I have been able to convert all the text to word vectors. I have generated two sets of word vectors, one using the tensor flow model, the other using gensim model. Both the models have been saved and are ready to be used or implemented on the recurrent neural networks. Please use the branch 'Word2VecTF' for the latest version.

For more details on the data set please follow the link:
https://www.kaggle.com/samdeeplearning/deepnlp

## Getting Started

The repositiory contains the data in dataset on the paths 'DeepNLP/DeepNLP/deepnlp/Sheet_1.csv' and 'DeepNLP/DeepNLP/deepnlp/Sheet_2.csv'. You can also get the dataset from the link
https://www.kaggle.com/samdeeplearning/deepnlp/data

### Prerequisites and Installation

The following code functions on Python3 having the libraries of tensor flow and gensim. Please follow the instructions from the below links:

> Python3: https://www.python.org/downloads/
> Tensor Flow: https://www.tensorflow.org/install/
> For gensim please follow the below command to execute on the terminal so that the library is in synchronization with the library that is used in the code.

```
pip install gensim==0.10.2
```

### Details of Code

The codebase consists of the below python files:

> load_preprocess_data.py: This code preproces data and spits out clean text in the path 'DeepNLP/DeepNLP/deepnlp/clean_chatbot.txt' and 'DeepNLP/DeepNLP/deepnlp/clean_resume.txt'. It also gives you statastics suc as the maximum words in a sentence and the maximum characters in a sentence of the whole dataset. The cleaned text is used as the main data for furter analyzing in other scripts.

> generate_word_vec.py: This a word2Vec model built on tensor flow. It is used to generate word vectors using SkipGram Model. These word vectors are then used in neural networks such as recurrent neural networks to build a robust machine learning algorithm.

> ml_rnn_model.py: This script currently generates word vectors using gensim library.

## Future Work

The Recurrent neural network will be implemented in the script ml_rnn_model.py. Tests will be run on the neural netwok to see how the model performs on different hyperparameters. Once the results are finalized, attempts will be made to combine all the scripts and deliver it as one package. In other words, running one script will load, preprocess, train, test and show useful statstics on data.

