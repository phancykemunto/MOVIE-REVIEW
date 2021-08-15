import numpy as np ## scientific computation
import pandas as pd ## loading dataset file
import nltk  ## Preprocessing Reviews
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords ## removing all the stop words
from gensim.utils import lemmatize
import re  ## To use Regular expression

#select data source
data= pd.read_csv('C:/PYDATAFILES/MOVIES.csv', encoding='latin')
#output the data container(frame)
print(data)

print(data.shape)  ### Return the shape of data 
print(data.ndim)   ### Return the n dimensions of data
print(data.size)   ### Return the size of data 
print(data.isna().sum())  ### Returns the sum fo all na values
print(data.info())  ### Give concise summary of a DataFrame
print(data.head())  ## top 5 rows of the dataframe
print(data.tail()) ## bottom 5 rows of the dataframe

#Cleaning The Dataset

corpus = []
for i in range(0,3000):   #we have 3000 reviews
     review = re.sub('[^a-zA-Z]'," ",data["review"][i])
     review = review.lower()
     review = review.split()
     pe = WordNetLemmatizer()
     all_stopword = stopwords.words('english')
     all_stopword.remove('not')
     
     review = " ".join(review)
     corpus.append(review)
print(corpus)

#Creating a Bage of words model for converting review into binary form
from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features=1500) ##1500 columns
cv = CountVectorizer()
X = cv.fit_transform(corpus)
y = data["sentiment"]

#Dumping Counter Verctorizer object for future use
import pickle
pickle.dump(cv, open('cv.pkl', 'wb'))

#spliting dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)


#Creating a model using MultinomialNB
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

filename = 'nlp_model.pkl'
pickle.dump(model, open(filename, 'wb'))
