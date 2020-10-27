# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:30:11 2020

@author: srinivasan.c
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import re
import nltk
corpus = []
dataset = pd.read_csv('data/Restaurant_Reviews.tsv', delimiter = '\t')
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
#import CountVectorizer from sklearn.
cv = CountVectorizer()#stop_words = 'english', lowercase = True, token_pattern = '[^a-zA-Z]')
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/5, random_state =0)
"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
"""
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(x_train,y_train)

y_pre =  classifier.predict(x_test)
corpus1 = []
review1 = 'Good One'
review1 = re.sub('[^a-zA-Z]', ' ', review1)
review1 = review1.lower()
review1 = review1.split()
ps = PorterStemmer()
review1 = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
review1 = ' '.join(review1)
corpus1.append(review1)
test_x = cv.fit_transform(corpus1).toarray()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pre)

print()