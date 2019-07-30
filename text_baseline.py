from sklearn import model_selection, preprocessing, linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import csv

#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers

csvfilename = 'riri145/post-data.csv'
trainDF = pd.read_csv(csvfilename, skiprows=21404, usecols=[0,2,7], names=['id', 'caption', 'ad'])

def cleanup(caption):
    return caption.replace('#ad', '').replace('#sponsored', '').replace('#Ad', '').replace('#advertisement', '')
trainDF['caption'] = trainDF['caption'].apply(cleanup)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['caption'], trainDF['ad'])
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['caption'])
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    return metrics.f1_score(predictions, valid_y)

f1 = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("R, Count Vectors: ", f1)
