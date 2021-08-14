import numpy as np ## scientific computation
import pandas as pd ## loading dataset file
#import matplotlib.pyplot as plt ## Visulization
#nltk.set_proxy('SYSTEM PROXY')
import nltk  ## Preprocessing Reviews
from nltk import WordNetLemmatizer
#nltk.download('stopwords') ##Downloading stopwords
from nltk.corpus import stopwords ## removing all the stop words
from gensim.utils import lemmatize
#from nltk.stem.porter import PorterStemmer ## stemming of words
import re  ## To use Regular expression

#select data source
data= pd.read_csv('C:/PYDATAFILES/MOVIES.csv', encoding='latin')
#output the shape of the data container(frame)
print(data)

print(data.shape)  ### Return the shape of data 
print(data.ndim)   ### Return the n dimensions of data
print(data.size)   ### Return the size of data 
print(data.isna().sum())  ### Returns the sum fo all na values
print(data.info())  ### Give concise summary of a DataFrame
print(data.head())  ## top 5 rows of the dataframe
print(data.tail()) ## bottom 5 rows of the dataframe

#import seaborn as sns
#sns.countplot('sentiment',data=data)

corpus = []
for i in range(0,3000):   #we have 1000 reviews
     review = re.sub('[^a-zA-Z]'," ",data["review"][i])
     review = review.lower()
     review = review.split()
     pe = WordNetLemmatizer()
     all_stopword = stopwords.words('english')
     all_stopword.remove('not')
     #review = [pe.lemmatize(word) for word in review if not word in set(all_stopword)]
     review = " ".join(review)
     corpus.append(review)
print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features=1500) ##1500 columns
cv = CountVectorizer()
#X = cv.fit_transform(corpus).toarray()
X = cv.fit_transform(corpus)
y = data["sentiment"]

import pickle
pickle.dump(cv, open('cv.pkl', 'wb'))

from sklearn.model_selection import train_test_split
#X = data.reshape(-1,3)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)
#spliting dataset into train and test
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 
#test size is 20% while 80% is the training data


from sklearn.naive_bayes import GaussianNB,MultinomialNB
GNB = GaussianNB()
MNB = MultinomialNB()

#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

filename = 'nlp_model.pkl'
pickle.dump(model, open(filename, 'wb'))
