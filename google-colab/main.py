"""
author:       sho

date:         2019-10-15

description:  multiclass classification of text inputs using keras. 
              code to train, evaluate, productionize and use model to predict in ~35 lines of code
              Three main sections:
              1) train and evaluate model
              2) productionize model (save the model and associated files)
              3) use the saved model to make predictions

references:   see below for a list of urls (resources) that were instrumental in helping with the creation of this code - thank you authors!

todo:         see below for full list of todo items
"""


import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.models import load_model
from keras import metrics
from keras import utils
from subprocess import check_output
import pickle
import helper as helper #only used if dataset is unbalanced or not optimized for ML

## get and prepare data

# load the dataset
df = pd.read_csv('dataTrain.txt')
#df = pd.read_csv('data/dataTrainSO.csv') #stack-overflow dataset
#X_train, y_train, X_test, y_test = helper.load_dataset('dataTrain.txt', 'text', 'label', 0.8)

print(df)

texts = df['text']
labels = df['label']
#texts = df['post'] #use this for the #stack-overflow dataset
#labels = df['tags'] #use this for the #stack-overflow dataset


# Split data into train and test (80% used for training, 20% for validation) (X = text Y = label)
train_size = int(len(df) * 0.8)
X_train = texts[:train_size]
y_train = labels[:train_size]
X_test = texts[train_size:]
y_test = labels[train_size:]

# prepare (encode) model input data (using tokenizer)
tok = Tokenizer()
tok.fit_on_texts(X_train)
X_train_enc = tok.texts_to_matrix(X_train, mode='count')
X_test_enc = tok.texts_to_matrix(X_test, mode='count')

# prepare (encode) target data (the labels that should be predicted)
num_classes = y_train.nunique()
le = LabelEncoder()
le.fit_transform(y_train)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test) #test labels transformed but not fit
y_train_enc_categorical = utils.to_categorical(y_train_enc, num_classes)
y_test_enc_categorical = utils.to_categorical(y_test_enc, num_classes)

#print("\n X_train_enc: \n", X_train_enc)
#print("\n X_test_enc: \n", X_test_enc)
#print("\n y_train_enc: \n", y_train_enc)
#print("\n y_test_enc: \n", y_test_enc)
#print("\n y_train_enc_categorical: \n", y_train_enc_categorical)
#print("\n y_test_enc_categorical: \n", y_test_enc_categorical)

#specify the model parameters
epochs=40
number_of_neurons = 500
batch_size = 128
dropout_fraction = 0.5
vocab_size = min(len(tok.word_index) + 1, 1000)

#define the model
model = Sequential()
model.add(Dense(number_of_neurons, input_shape=(vocab_size,), activation='relu'))
model.add(Dropout(dropout_fraction, seed=None))
model.add(Dense(num_classes)) 
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
print(model.summary())

# fit the model
model.fit(X_train_enc, y_train_enc_categorical, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

# Evaluate the accuracy of the trained model
score = model.evaluate(X_test_enc, y_test_enc_categorical, batch_size=batch_size, verbose=1)
print("score: ", score)

## Productionize the model

#save the model (so that it can be used later)
model.save('model.h5')

#save/Pickle the tokenizer
with open('tokenizer.pickle', 'wb') as handle: pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

#save/Pickle the Label Encoder
with open('labelencoder.pickle', 'wb') as handle: pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)


## Use the saved model to make predictions

#load the model
model = load_model('model.h5')

#load the label encoder
with open('labelencoder.pickle', 'rb') as handle: le = pickle.load(handle)

# load the tokenizer
with open('tokenizer.pickle', 'rb') as handle: tok = pickle.load(handle)


#create a dataframe from the test datafile
df = pd.read_csv('dataTest.txt')
#df = pd.read_csv('data/dataTestSO.txt') #stack-overflow dataset

#create the output dataframe
df_output = pd.DataFrame(columns=['text', 'label'])

#encode the text values we want to predict on using the loaded tokenizer pickle file
X_test = df['text']
X_test_enc = tok.texts_to_matrix(X_test, mode='count')

text_labels = le.classes_
for i in range(0,X_test_enc.shape[0]):
  prediction = model.predict(np.array([X_test_enc[i]]))
  predicted_label = text_labels[np.argmax(prediction)]
  #print("input: ", X_test[i])
  #print("prediction: ", prediction)
  #print("predicted_label: ", predicted_label)
  df_output = df_output.append({'text': X_test.iloc[i], 'label':predicted_label}, ignore_index=True)

print (df_output)
