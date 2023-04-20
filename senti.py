#!/usr/bin/env python
# coding: utf-8


import re
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
import nltk
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

def sentiment_analysis(test):
    train = pd.read_csv('./data/train.csv')
    train = train[['selected_text','sentiment']]
    train["selected_text"].fillna("No content", inplace = True)

    def depure_data(data):
        
        #Removing URLs with a regular expression
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        data = url_pattern.sub(r'', data)

        # Remove Emails
        data = re.sub('\S*@\S*\s?', '', data)

        # Remove new line characters
        data = re.sub('\s+', ' ', data)

        # Remove distracting single quotes
        data = re.sub("\'", "", data)
            
        return data


    temp = []
    #Splitting pd.Series to list
    data_to_list = train['selected_text'].values.tolist()
    for i in range(len(data_to_list)):
        temp.append(depure_data(data_to_list[i]))


    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
            

    data_words = list(sent_to_words(temp))

    def detokenize(text):
        return TreebankWordDetokenizer().detokenize(text)


    data = []
    for i in range(len(data_words)):
        data.append(detokenize(data_words[i]))

    data = np.array(data)


    from keras_preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences

    max_words = 5000
    max_len = 200

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)

    import keras.models
    best_model = keras.models.load_model("Sentiment_analysis_BiLSTM.hdf5")

    sentiment = [0,-1,1]

    sent=[]
    for i in test['message']:
        sequence = tokenizer.texts_to_sequences([i])
        sentest = pad_sequences(sequence, maxlen=200)
        sent.append(sentiment[np.around(best_model.predict(sentest), decimals=0).argmax(axis=1)[0]])

    test['value']=sent
    
    return test




