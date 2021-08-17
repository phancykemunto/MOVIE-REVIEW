import numpy as np ## scientific computation
import pandas as pd ## loading dataset file
##import matplotlib.pyplot as plt ## Visulization

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
#sns.countplot('Sentiment',data=data)

import nltk  ## Preprocessing Reviews
nltk.download('stopwords') ##Downloading stopwords
nltk.download('wordnet')
from nltk.corpus import stopwords ## removing all the stop words
from nltk.stem.porter import PorterStemmer ## stemming of words
from nltk.stem import WordNetLemmatizer
import re  ## To use Regular expression
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(0,3000):   #we have 3000 reviews
     review = re.sub('[^a-zA-Z]'," ",data["Review"][i])
     review = review.lower()
     review = review.split()
     pe = PorterStemmer()
     all_stopword = stopwords.words('english')
     all_stopword.remove('not')
    #remove negative word 'not' as it is closest word to help determine whether the review is good or not 
     review = [pe.stem(word) for word in review if not word in set(all_stopword)]
     review = " ".join(review)
     corpus.append(review)
print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) ##1500 columns
X = cv.fit_transform(corpus).toarray()

y = data["Sentiment"]

import pickle
# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('countvector.pkl', 'wb'))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)


#MultinomialNB
from sklearn.naive_bayes import MultinomialNB,GaussianNB
classifier = GaussianNB().fit(X_train, y_train)
MNB = MultinomialNB()
cls = MultinomialNB().fit(X_train, y_train)

cls.score(X_test,y_test)

classifier.score(X_test,y_test)

# Creating a pickle file for the Multinomial Naive Bayes model
#filename = 'voting_clf.pkl'
pickle.dump(cls, open("Review.pkl", 'wb'))