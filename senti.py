import warnings
warnings.filterwarnings('ignore')
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras.models
max_words = 5000
max_len = 200
def sentiment_analysis(test):
    data=np.load('data.npy')
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    best_model = keras.models.load_model("Sentiment_analysis_BiLSTM.hdf5")
    sentiment = [0,-1,1]
    sent=[]
    for i in test['message']:
        sequence = tokenizer.texts_to_sequences([i])
        sentest = pad_sequences(sequence, maxlen=200)
        sent.append(sentiment[np.around(best_model.predict(sentest), decimals=0).argmax(axis=1)[0]])
    test['value']=sent
    return test