from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import re
import pandas as pd 
from nltk.corpus import stopwords
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import HTMLConverter,TextConverter,XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
import os
import sys, getopt
import pickle

data = pd.read_csv("C:/Users/hind/Desktop/start_up/web_scrapping/cv_dataset1.csv",encoding = "utf-8")
data = data.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;•""]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
english = set(stopwords.words('english'))
french=set(stopwords.words('french'))
STOPWORDS=english.union(french)

def clean_text(text):
    
    text = text.lower()
   #text = re.sub('http\S+\s*', ' ', text)  # remove URLs
   #text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    #text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text

   

    return text

data_test=data['competences']+' '+data['diplome']+' '+data['secteur_experience']+' '+data['establisment']
data_test=data_test.str.replace('None', '')

data_test=data_test.astype(str)
data_test=data_test.apply(clean_text)


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data_test, data.metiers_recherches)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(data_test)

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(data_test)
transformer=tfidf_vect
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
# Linear Classifier on Word Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy) 

model = linear_model.LogisticRegression().fit(xtrain_tfidf, train_y)

model_path = 'model.pkl'
transformer_path='transformer.pkl'
pickle.dump(model, open(model_path, 'wb'))
pickle.dump(transformer,open(transformer_path,'wb'))


with open('C:/Users/hind/Desktop/start_up/corpus/AgtayMeryem.txt', 'r') as file:
    data = file.read().replace('\n', '')

print(model.predict(transformer.transform([data])))
