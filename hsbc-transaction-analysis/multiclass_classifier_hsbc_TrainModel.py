#Author:  Sho Munshi
#Desc:    First ever model! 
#         Pass in hsbc transaction data (statement transactions & category labels)
#         to train model to predict the category labels of statement transactions
#Date:    2019-02-28 (Completed)


#####################################################################################
# Most of this code (except for knowning to save the LabelEncoder and Tokenizer) was 
# written thanks to the authors of the below sites:

#https://www.kaggle.com/jacklinggu/keras-mlp-cnn-test-for-text-classification/data
#https://github.com/keras-team/keras/tree/master/keras
#https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance
#https://github.com/keras-team/keras/blob/master/keras/metrics.py
#https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
#https://machinelearningmastery.com/what-are-word-embeddings/
#https://github.com/scikit-learn-contrib/categorical-encoding/issues/26
#####################################################################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
import time
from keras import metrics

from subprocess import check_output
from keras import utils

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import itertools

import pickle
#####################################################################################
#Shows list of available files in the directory
#print(check_output(["ls", "../_data"]).decode("utf8"))
#####################################################################################
def f_train_model(
    p_input_file, 
    p_input_file_text_column,
    p_input_file_tag_column,
    p_model_file, 
    p_tokenizer_file, 
    p_labelencoder_file,
    p_batch_size,
    p_epochs,
    p_dict_size,
    p_train_size_percentage
):
    
    print('*********************| ' + p_input_file_tag_column + '|*********************')
    
    data_file = p_input_file
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    texts = df[p_input_file_text_column]
    tags = df[p_input_file_tag_column]
    tags = tags.fillna('**UNKNOWN**')  #REPLACE NAN WITH **UNKNOWN**
    #tags = tags.str.replace('0')  #REPLACE NAN WITH **UNKNOWN** 
    
    print(tags.unique())
    
    # Split data into train and test
    #train_size = int(len(df) * .8)
    train_size = int(len(df) * p_train_size_percentage)
    
    print ("Train size: %d" % train_size)
    print ("Test size: %d" % (len(df) - train_size))

    train_texts = texts[:train_size]
    train_tags = tags[:train_size]

    test_texts = texts[train_size:]
    test_tags = tags[train_size:]

        
    print('*********************| ' + '|*********************')
    
    #show differences between the tags in train and test (if any)
    x = train_tags.drop_duplicates()
    y = test_tags.drop_duplicates()
    print(x[~x.isin(y)])
    print(y[~y.isin(x)])
    
    
    
    num_max = p_dict_size #this is usually a lot lower if it's just for real english words

    # preprocess
    encoder = LabelEncoder()
    train_tags_encoded = encoder.fit_transform(train_tags) #fit AND transform in one function - I think

    num_classes = np.max(train_tags_encoded) + 1

    test_tags_encoded = encoder.transform(test_tags) #test tags transformed but not fit
    test_tags_encoded_to_categorical = utils.to_categorical(test_tags_encoded, num_classes)

    tok = Tokenizer(num_words=num_max)
    tok.fit_on_texts(train_texts)
    train_texts_matrix = tok.texts_to_matrix(train_texts, mode='count')
    test_texts_matrix = tok.texts_to_matrix(test_texts, mode='count')
    
    #batch size is used later to evaluate the model 
    # and also used in building the model so setting it here
    batch_size = p_batch_size
    epochs = p_epochs

    
    # (multi-class classification model (categorical_crossentropy))
    #define the model    
    model = Sequential()
    model.add(Dense(512, input_shape=(num_max,)))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[metrics.categorical_accuracy]) #, metrics.binary_accuracy, metrics.mean_squared_error, metrics.mse, metrics.msle])

    train_tags_encoded_to_catagorical = utils.to_categorical(train_tags_encoded, num_classes)

    #fit the model with train data
    model.fit(train_texts_matrix, train_tags_encoded_to_catagorical, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)
    
    
    # Evaluate the accuracy of the trained model
    score = model.evaluate(test_texts_matrix, test_tags_encoded_to_categorical, batch_size=batch_size, verbose=1)
    
    print(model.metrics_names)
    print('Test score (loss):', score[0])
    print('Test accuracy (accuracy):', score[1])
    
    
    text_labels = encoder.classes_ 
    print (text_labels)
    
    #this calculates the values required for the confusion matrix using the test data
    y_softmax = model.predict(test_texts_matrix)

    y_test_1d = []
    y_pred_1d = []

    for i in range(len(test_tags_encoded_to_categorical)):
        probs = test_tags_encoded_to_categorical[i]
        index_arr = np.nonzero(probs)
        one_hot_index = index_arr[0].item(0)
        y_test_1d.append(one_hot_index)

    for i in range(0, len(y_softmax)):
        probs = y_softmax[i]
        predicted_index = np.argmax(probs)
        y_pred_1d.append(predicted_index)
        
    
    #plot the confusion matrix
    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    plt.figure(figsize=(24,20))
    plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
    plt.show()


    #Save the model (so that it can be used outside of this notebook)
    model.save(p_model_file)

    #Save the tokenizer - you need this when you fit the data you want to 
    #    predict for and is not saved as part of the model (or I couldn't figure out how)
    with open(p_tokenizer_file, 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Save the Label Encoder - as when the model predicts it does not know the labels
    #    the label encoding is not saved as part of the model (or I couldn't figure out how)
    with open(p_labelencoder_file, 'wb') as handle:
        pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
        
        
# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)

#####################################################################################
#VARIABLES (First 2 need to be changed only)
train_file = '../_data/hsbc/TRAINING_FILE_20190509.csv'
file_date_string = '20190509'
batch_size = 32
epochs = 2
dict_size = 10000
train_size_percentage = 0.99 #this should be 0.8 but I dont care to train and test for production


#category
f_train_model(
    train_file,
    'original_description',
    'category_model_and_manual',
    '../_model/multiclass_classifier_hsbc_Categories_' + file_date_string + '.h5',
    '../_model/tokenizer_hsbc_Categories_' + file_date_string + '.pickle',
    '../_model/labelencoder_hsbc_Categories_' + file_date_string + '.pickle',
    batch_size,
    epochs,
    dict_size,
    train_size_percentage
)

#subcategory
f_train_model(
    train_file,
    'original_description',
    'subcategory_model_and_manual',
    '../_model/multiclass_classifier_hsbc_SubCategories_' + file_date_string + '.h5',
    '../_model/tokenizer_hsbc_SubCategories_' + file_date_string + '.pickle',
    '../_model/labelencoder_hsbc_SubCategories_' + file_date_string + '.pickle',
    batch_size,
    epochs,
    dict_size,
    train_size_percentage
)

#provider
f_train_model(
    train_file,
    'original_description',
    'uprovider',
    '../_model/multiclass_classifier_hsbc_Provider_' + file_date_string + '.h5',
    '../_model/tokenizer_hsbc_Provider_' + file_date_string + '.pickle',
    '../_model/labelencoder_hsbc_Provider_' + file_date_string + '.pickle',
    batch_size,
    epochs,
    dict_size,
    train_size_percentage
)

#tag01
f_train_model(
    train_file,
    'original_description',
    'utag01',
    '../_model/multiclass_classifier_hsbc_Tag01_' + file_date_string + '.h5',
    '../_model/tokenizer_hsbc_Tag01_' + file_date_string + '.pickle',
    '../_model/labelencoder_hsbc_Tag01_' + file_date_string + '.pickle',
    batch_size,
    epochs,
    dict_size,
    train_size_percentage
)

#tag02
f_train_model(
    train_file,
    'original_description',
    'utag02',
    '../_model/multiclass_classifier_hsbc_Tag02_' + file_date_string + '.h5',
    '../_model/tokenizer_hsbc_Tag02_' + file_date_string + '.pickle',
    '../_model/labelencoder_hsbc_Tag02_' + file_date_string + '.pickle',
    batch_size,
    epochs,
    dict_size,
    train_size_percentage
)
#####################################################################################
#####################EVERYTHING BELOW HERE IS FOR DEBUG PURPOSES##################### 
#####################################################################################
data_file = '../_data/hsbc/ACCOUNTS_2019_20190314.csv'
df = pd.read_csv(data_file)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

print(df.head())

texts = df.original_description

tags = df.category_model_and_manual
tags = tags.fillna('**UNKNOWN**')  #REPLACE NAN WITH **UNKNOWN**
#tags = tags.str.replace('0')  #REPLACE NAN WITH **UNKNOWN**
#####################################################################################
print(tags.unique())
#####################################################################################
# Split data into train and test
train_size = int(len(df) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(df) - train_size))
#####################################################################################
train_texts = texts[:train_size]
train_tags = tags[:train_size]

test_texts = texts[train_size:]
test_tags = tags[train_size:]

print(train_texts.shape)
print(train_tags.shape)
print(test_texts.shape)
print(test_tags.shape)
#####################################################################################
#check that all the tags/categories in test and train are the same
#(otherwise: the convolution matrix will not be right
#            and if you have more in test that train then you havent trained those items 
#                   maybe alos yo'll get an error!))

print(train_tags.nunique())
print(test_tags.nunique())

print(train_tags.value_counts().sort_index())
print(test_tags.value_counts().sort_index())

#####################################################################################
#show differences between the tags in train and test (if any)

x = train_tags.drop_duplicates()
y = test_tags.drop_duplicates()

print(x[~x.isin(y)])
print(y[~y.isin(x)])
#####################################################################################
num_max = 10000 #this is usually a lot lower if it's just for real english words

# preprocess
encoder = LabelEncoder()
train_tags_encoded = encoder.fit_transform(train_tags) #fit AND transform in one function - I think

num_classes = np.max(train_tags_encoded) + 1

test_tags_encoded = encoder.transform(test_tags) #test tags transformed but not fit
test_tags_encoded_to_categorical = utils.to_categorical(test_tags_encoded, num_classes)

tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(train_texts)
train_texts_matrix = tok.texts_to_matrix(train_texts, mode='count')
test_texts_matrix = tok.texts_to_matrix(test_texts, mode='count')
#####################################################################################

#print(tok.word_counts)
#print(tok.document_count)
#print(tok.word_index)
#print(tok.word_docs)

#word_counts: A dictionary of words and their counts.
#word_docs: A dictionary of words and how many documents each appeared in.
#word_index: A dictionary of words and their uniquely assigned integers.
#document_count: An integer count of the total number of documents that were used to fit the Tokenizer.
#####################################################################################
#batch size is used later to evaluate the model 
# and also used in building the model so setting it here
batch_size = 32
epochs = 5

#####################################################################################
# (multi-class classification model (categorical_crossentropy))

#define the model    
model = Sequential()
model.add(Dense(512, input_shape=(num_max,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[metrics.categorical_accuracy]) #, metrics.binary_accuracy, metrics.mean_squared_error, metrics.mse, metrics.msle])

train_tags_encoded_to_catagorical = utils.to_categorical(train_tags_encoded, num_classes)

#fit the model with train data
model.fit(train_texts_matrix, train_tags_encoded_to_catagorical, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

#####################################################################################

# Evaluate the accuracy of the trained model
score = model.evaluate(test_texts_matrix, test_tags_encoded_to_categorical, batch_size=batch_size, verbose=1)

print(model.metrics_names)
print('Test score (loss):', score[0])
print('Test accuracy (accuracy):', score[1])
#####################################################################################

text_labels = encoder.classes_ 
print (text_labels)

#####################################################################################

# Here's how to generate a prediction on individual examples from the test data

for i in range(30,50):
    prediction = model.predict(np.array([test_texts_matrix[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    
    print('Text Input: ' + test_texts.iloc[i][:50], "...")
    print('Actual Label: ' + test_tags.iloc[i])
    print('Predicted Label: ' + predicted_label + "\n")
    #print(test_texts_matrix[i]) # read this in conjunction with text_labels and tok.word_index
    #print("Predicted Label Index: " + np.argmax(prediction))

#####################################################################################

#this calculates the values required for the confusion matrix using the test data

y_softmax = model.predict(test_texts_matrix)

y_test_1d = []
y_pred_1d = []

for i in range(len(test_tags_encoded_to_categorical)):
    probs = test_tags_encoded_to_categorical[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)

#####################################################################################

# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)
#####################################################################################

cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(24,20))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
plt.show()
#####################################################################################

#Save the model (so that it can be used outside of this notebook)
model.save('../_model/multiclass_classifier_hsbc_Categories_20190320.h5')

#Save the tokenizer - you need this when you fit the data you want to 
#    predict for and is not saved as part of the model (or I couldn't figure out how)
with open('../_model/tokenizer_hsbc_Categories_20190320.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Save the Label Encoder - as when the model predicts it does not know the labels
#    the label encoding is not saved as part of the model (or I couldn't figure out how)
with open('../_model/labelencoder_hsbc_Categories_20190320.pickle', 'wb') as handle:
    pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)


#####################################################################################
# see notebook "multiclass_classifier_hsbc_production" to learn how to call the saved model to:
# 1) pass in 1 input to get a prediction
# 2) pass in many inputs to get predictions
#####################################################################################

#####################################################################################


#####################################################################################
