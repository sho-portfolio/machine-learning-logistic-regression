import _helper as helper
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
import tensorflow as tf
import pickle

def f_train_model(
  p_model_optimizer, #Adam, SGD
  p_training_data_file,
  p_training_data_file_text_column,
  p_training_data_file_label_column,
  p_model_file,
  p_tokenizer_file,
  p_labelencoder_file,
  p_batch_size,
  p_epochs,
  p_dict_size,
  p_train_size_percentage,
  p_plot_confusion_matrix,
  p_number_of_neurons,
  p_use_dropout,
  p_dropout_fraction
  ):

    df = pd.read_csv(p_training_data_file)
    print(df)

    # added this as if I pass the entire dataframe I get this error in repl (repl process died unexpectedly: signal: killed).  Probably becuase it doesnt have enough memory.  With this code I simply only keep the columns in the dataframe that are needed
    df = df[[p_training_data_file_text_column, p_training_data_file_label_column]].copy()
    
    # this codes takes the dataset and to prevent the unseen error* takes the first instance of every label in the test data and adds it to the train data
    #* ValueError: ValueError: y contains previously unseen labels: 'CIGS'
    #* The error comes from this line of code:     test_labels_encoded = encoder.transform(test_labels)
    ddTrain, ddTest = helper.IdentifyNonMatchingValuesBetweenDataSetsv2(df, p_train_size_percentage, p_training_data_file_label_column, p_training_data_file_text_column)
    df = pd.concat([ddTrain, ddTest])

    texts = df[p_training_data_file_text_column]
    labels = df[p_training_data_file_label_column]
    labels = labels.fillna('**UNKNOWN**')  #REPLACE NAN WITH **UNKNOWN**
    #labels = labels.str.replace('0')  #REPLACE NAN WITH **UNKNOWN**
    
    print(labels.unique())
    
    # Split data into train and test
    train_size = int(len(df) * p_train_size_percentage)
    print ("Train size: %d" % train_size)
    print ("Test size: %d" % (len(df) - train_size))
    
    train_texts = texts[:train_size]
    train_labels = labels[:train_size]
    
    test_texts = texts[train_size:]
    test_labels = labels[train_size:]
    
    #show differences between the labels in train and test (if any)
    x = train_labels.drop_duplicates()
    y = test_labels.drop_duplicates()
    print(x[~x.isin(y)])
    print(y[~y.isin(x)]) #any output here will cause problems but should not happen thanks to the IdentifyNonMatchingValuesBetweenDataSetsv2 function
        
    # one hot encode the classes (ie the values we are trying to predict) using sci-kit learn
    encoder = LabelEncoder()
    train_labels_encoded = encoder.fit_transform(train_labels) #fit AND transform in one function - I think
    
    num_classes = np.max(train_labels_encoded) + 1
    
    test_labels_encoded = encoder.transform(test_labels) #test labels transformed but not fit
    test_labels_encoded_to_categorical = utils.to_categorical(test_labels_encoded, num_classes)
    tok = Tokenizer(num_words=p_dict_size)
    tok.fit_on_texts(train_texts)
    train_texts_matrix = tok.texts_to_matrix(train_texts, mode='count')
    test_texts_matrix = tok.texts_to_matrix(test_texts, mode='count')
  
    train_labels_encoded_to_catagorical = utils.to_categorical(train_labels_encoded, num_classes)

    # (multi-class classification model (categorical_crossentropy))
    #define the model    
    model = create_baseline_model(
      p_model_optimizer = p_model_optimizer,
      dictionary_size = p_dict_size,
      number_of_neurons = p_number_of_neurons,
      use_dropout = p_use_dropout,
      dropout_fraction = p_dropout_fraction, #between 0 and 1,
      number_of_classes = num_classes
    )
  
    #fit the model with train data | shuffle=Flase seems to give us more consistent accuracy scores between runs of the same data & parameters (but not 100% sure yet)
    model.fit(train_texts_matrix, train_labels_encoded_to_catagorical,
      batch_size=p_batch_size, 
      epochs=p_epochs, 
      verbose=1, 
      validation_split=0.2, 
      shuffle=False)
     
    # Evaluate the accuracy of the trained model
    score = model.evaluate(
      test_texts_matrix, 
      test_labels_encoded_to_categorical, 
      batch_size=p_batch_size, 
      verbose=1)

    print("****** MODEL INFO ******")
    print (model)
    print(model.metrics_names)
    print(model.optimizer)
    print('Test score (loss):', score[0])
    print('Test score (accuracy):', score[1])
    #print(model.outputshape)
    print(model.summary())
    print(model.get_config())
    print(model.get_weights())
    print("************************")


    # Write model evaluation info to log file for later analysis
    helper.write_log_file(p_training_data_file_label_column, p_batch_size, p_epochs, p_dict_size, str(model.metrics_names), str(score[0]), str(score[1]), p_train_size_percentage, model.optimizer, model.loss, p_dict_size, p_number_of_neurons, p_use_dropout, p_dropout_fraction, num_classes, train_size)


    #todo: add back code for displaying confusion matrix

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

def create_baseline_model(
  p_model_optimizer,
  dictionary_size,
  number_of_neurons,
  use_dropout,
  dropout_fraction, #between 0 and 1,
  number_of_classes
  ):
  # (multi-class classification model (categorical_crossentropy))
  #define the model
  # 1) instantiate a Sequential Model
  # 2) add layers to it one by one using Add
  # 3) compile the model with a loss function, an optimizer and optional evealuation metrics
  # 4) use data to fit the model
  # 5) evaluate model, persist or deploy model

  model = Sequential()
  
  model.add(Dense(
    number_of_neurons, 
    input_shape=(dictionary_size,), #Input shape only needs to de fined for the first layer
    activation='relu'))
  
  if use_dropout:
    #model.add(Dropout(dropout_fraction))
    model.add(Dropout(
      dropout_fraction, #Fraction of units to drop 
      seed=None         #Random seed for reporducibility
      ))

  model.add(Dense(number_of_classes))
  model.add(Activation('softmax'))

  """
  #Method 1: hardcoded loss and optimizer
  model.compile(loss='categorical_crossentropy',
                optimizer=p_model_optimizer,
                metrics=[metrics.categorical_accuracy]) #, metrics.binary_accuracy, metrics.mean_squared_error, metrics.mse, metrics.msle])
  """
  #Method 2: loss and optimizer imported from keras modules (preferred)
  from keras.losses import categorical_crossentropy
  from keras import optimizers
  sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

  if p_model_optimizer == 'SGD':
    optimizer = sgd
  elif p_model_optimizer == 'Adam':
    optimizer = adam
  else:
    print("****OPTIMIZER NOT DEFINED****")

  model.compile(loss=categorical_crossentropy,
                optimizer=optimizer,
                metrics=[metrics.categorical_accuracy]) #, metrics.binary_accuracy, metrics.mean_squared_error, metrics.mse, metrics.msle])


  return model

  #original code...

  #model = Sequential()
  #model.add(Dense(1000, input_shape=(p_dict_size,),activation='relu'))
  #model.add(Dropout(0.5))
  
  #model.add(Dense(num_classes))
  #model.add(Activation('softmax'))
  
  #model.compile(loss='categorical_crossentropy',
  #              optimizer=p_model_optimizer,
  #              metrics=[metrics.categorical_accuracy]) #, metrics.binary_accuracy, #metrics.mean_squared_error, metrics.mse, metrics.msle])
