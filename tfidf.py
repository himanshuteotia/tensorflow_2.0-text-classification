#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:10:43 2020

@author: himanshu teotia
"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'



"""
Machine learning algorithms cannot work with raw text directly. Rather, the text must be converted into vectors of numbers. 
In natural language processing, a common technique for extracting features from text is to place all of the words that occur 
in the text in a bucket. This aproach is called a bag of words model or BoW for short. It’s referred to as a “bag” of words 
because any information about the structure of the sentence is lost.

"""
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')

"""
By casting the bag of words to a set, we can automatically remove any duplicate words.

"""

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))


"""
Next, we’ll create a dictionary of words and their occurence for each document in the corpus (collection of documents).

"""

numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1
    
    
"""
Another problem with the bag of words approach is that it doesn’t account for noise. In other words, certain words are used 
to formulate sentences but do not add any semantic meaning to the text. For example, the most commonly used word in the 
english language is the which represents 7% of all words written or spoken. You couldn’t make deduce anything about a text 
given the fact that it contains the word the. On the other hand, words like good and awesome could be used to determine 
whether a rating was positive or not.

In natural language processing, useless words are referred to as stop words. The python natural language toolkit library
 provides a list of english stop words.

"""


from nltk.corpus import stopwords
stopwords.words('english')


"""
Often times, when building a model with the goal of understanding text, you’ll see all of stop words being removed. 

Another strategy is to score the relative importance of words using TF-IDF.

"""


"""
Term Frequency (TF)
The number of times a word appears in a document divded by the total number of words in the document. 
Every document has its own term frequency

"""


def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

#The following lines compute the term frequency for each of our documents.
tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)


#The following code implements inverse data frequency in python.

"""
Inverse Data Frequency (IDF)

The log of the number of documents divided by the number of documents that contain the word w. 
Inverse data frequency determines the weight of rare words across all documents in the corpus.

"""

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict



#The IDF is computed once for all documents.

idfs = computeIDF([numOfWordsA, numOfWordsB])


"""

Lastly, the TF-IDF is simply the TF multiplied by IDF.

"""

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


#Finally, we can compute the TF-IDF scores for all the words in the corpus.
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
df = pd.DataFrame([tfidfA, tfidfB])



"""
Rather than manually implementing TF-IDF ourselves, we could use the class provided by sklearn.

"""

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([documentA, documentB])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)


# reference doc https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76

