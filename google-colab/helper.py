
import pandas as pd
from keras.preprocessing.text import Tokenizer

def load_dataset(p_training_data_file, p_training_data_file_text_column, p_training_data_file_label_column, p_train_size_percentage):
  df = pd.read_csv(p_training_data_file)
  print(df)

  #clean the dataset (hsbc)
  #df = cleanHsbcData(df, p_training_data_file_text_column)


  df = df[[p_training_data_file_text_column, p_training_data_file_label_column]].copy()

  ddTrain, ddTest = IdentifyNonMatchingValuesBetweenDataSetsv2(df, p_train_size_percentage, p_training_data_file_label_column, p_training_data_file_text_column)
  df = pd.concat([ddTrain, ddTest])

  texts = df[p_training_data_file_text_column]
  labels = df[p_training_data_file_label_column]
  labels = labels.fillna('**UNKNOWN**')  #REPLACE NAN WITH **UNKNOWN**
  #labels = labels.str.replace('0')  #REPLACE NAN WITH **UNKNOWN**
  
  print("labels: ", labels.unique())
  
  # Split data into train and test
  train_size = int(len(df) * p_train_size_percentage)
  print ("Train size: %d" % train_size)
  print ("Test size: %d" % (len(df) - train_size))
  
  train_texts = texts[:train_size]
  train_labels = labels[:train_size]
  
  test_texts = texts[train_size:]
  test_labels = labels[train_size:]
  
  #show differences between the labels in train and test (if any)
  train_labels_nodupe = train_labels.drop_duplicates()
  test_labels_nodupe = test_labels.drop_duplicates()
  print(train_labels_nodupe[~train_labels_nodupe.isin(test_labels_nodupe)])
  print(test_labels_nodupe[~test_labels_nodupe.isin(train_labels_nodupe)]) 
  #any output here will cause problems but should not happen thanks to the IdentifyNonMatchingValuesBetweenDataSetsv2 function

  #y_train_unique_cnt = len(train_labels_nodupe)
  #y_test_unique_cnt = len(test_labels_nodupe)

  X_train = train_texts
  y_train = train_labels

  X_test = test_texts
  y_test = test_labels

  return X_train, y_train, X_test, y_test


def IdentifyNonMatchingValuesBetweenDataSetsv2(df, TrainSizePercentage, tag_col, text_col):
    
    #tag_col = 'subcategory_model_and_manual'
    #text_col = 'original_description'
    
    TrainSize = int(len(df) * TrainSizePercentage)

    print(TrainSize, TrainSizePercentage)
    
    dfTrain = df[:TrainSize]
    dfTest = df[TrainSize:]

    #left join test and train datasets and return only values from test that do not appear in train (to avoid the unseen error)
    df_test_not_in_train = pd.merge(dfTest, dfTrain, on=[tag_col], how="left", indicator=True)

    df_test_not_in_train = df_test_not_in_train.query('_merge == "left_only"')

    #keep only the first occurence of a tag
    df_test_not_in_train = df_test_not_in_train.drop_duplicates([tag_col], keep="first")

    #get rid of system columns created during merge rename the columns so that when you concat later the column names line up)
    df_test_not_in_train = df_test_not_in_train[[text_col+ "_x", tag_col]]
    df_test_not_in_train.columns = [text_col, tag_col]

    #concat the values in test but not in train to the training dataset
    dfTrain = pd.concat([df_test_not_in_train, dfTrain], sort=False)


    print("original train size: " + str(TrainSize))
    print("new train size: " + str(len(dfTrain)))

    return dfTrain, dfTest


def getvocabsize(X_train):
  tok = Tokenizer()
  tok.fit_on_texts(X_train)
  return(len(tok.word_index) + 1)
