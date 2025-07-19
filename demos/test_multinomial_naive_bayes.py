import numpy as np
import pandas as pd
import sklearn
import nltk
from nltk.corpus import stopwords

import string
import os

from naive_bayes import MultiNaiveBayes


def get_data():
    print("Getting data...")
    newstrain = sklearn.datasets.fetch_20newsgroups(subset='train')
    newstest = sklearn.datasets.fetch_20newsgroups(subset='test')
    nltk.download('stopwords')

    myStopWords = list(string.punctuation) + stopwords.words('english')
    CV = sklearn.feature_extraction.text.CountVectorizer(max_features=3000,stop_words=myStopWords)

    xtrain = CV.fit_transform(newstrain.data)
    xtrain = xtrain.toarray()
    ytrain = newstrain.target

    xtest = CV.fit_transform(newstest.data)
    xtest = xtest.toarray()
    ytest = newstest.target

    return xtrain, ytrain, xtest, ytest

def main():
    xtrain, ytrain, xtest, ytest = get_data()
    
    nb = MultiNaiveBayes()
    nb.fit(xtrain, ytrain)
    
    train_res = nb.batch_predict(xtrain)
    test_res = nb.batch_predict(xtest)
    print(f"Training accuracy: {np.sum(train_res == ytrain) / ytrain.shape[0]}")
    print(f"Testing accuracy: {np.sum(test_res == ytest) / ytest.shape[0]}")


if __name__ == "__main__":
    main()
