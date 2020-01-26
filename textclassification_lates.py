

# all the imports 
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


# data url here 
# https://github.com/himanshuteotia/intent-classification/blob/master/training_phrases.csv



df = pd.read_csv("training_phrases.csv")
df = df[pd.notnull(df['intents'])]
df['phrases'].apply(lambda x: len(x.split(' '))).sum()
my_tags = list(set(df['intents']))
plt.figure(figsize=(10,4))
df.intents.value_counts().plot(kind='bar');


def print_plot(index):
    example = df[df.index == index][['phrases', 'intents']].values[0]
    if len(example) > 0:
        print('phrase : ',example[0])
        print('intent : ', example[1])


print_plot(12)


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "html.parser").get_text() # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

df['phrases'] = df['phrases'].apply(clean_text)

print(df['phrases'][0:20])


df['phrases'].apply(lambda x: len(x.split(' '))).sum()

# Now we have over 1434 words to work with.


#from sklearn.utils import shuffle
#df = shuffle(df)
X = df.phrases
y = df.intents

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state = 42)

"""

The next steps includes feature engineering. We will convert our text documents to a matrix of token counts (CountVectorizer), 
then transform a count matrix to a normalized tf-idf representation (tf-idf transformer). After that, we train several classifiers.

"""

# Naive Bayes classifier for multinomial models

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)


from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test)) # very bad accouracy accuracy 0.3968609865470852
#print(classification_report(y_test, y_pred,target_names=my_tags))


print(my_tags)

print(set(df.intents))

print(nb.predict(["old"]))


# Linear support vector machine

from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)


y_pred_y = sgd.predict(X_test)

print('accuracy ok sgd %s' % accuracy_score(y_pred_y, y_test))
#print(classification_report(y_test, y_pred,target_names=my_tags))


#Logistic regression

from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)


y_pred_q = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred_q, y_test))


# Word2vec embedding and Logistic Regression

from gensim.models import Word2Vec

wv = gensim.models.KeyedVectors.load_word2vec_format("/home/himanshu/Documents/NLP_PROECTS/word_embeddings/GoogleNews-vectors-negative300.bin", binary=True)
wv.init_sims(replace=True)


from itertools import islice
list(islice(wv.vocab, 13030, 13050))

"""
The common way is to average the two word vectors. BOW based approaches which includes averaging.
"""


def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


train, test = train_test_split(df, test_size=0.1, random_state = 42)

test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['phrases']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['phrases']), axis=1).values

X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_word_average, train['intents'])
y_pred = logreg.predict(X_test_word_average)


print('accuracy %s' % accuracy_score(y_pred, test.intents))









