import pandas as pd
import os
import datetime
#%matplotlib inline. ##for some reason this does not work in repl.it
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


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



def write_log_file(
  p_input_file_tag_column,
  p_batch_size,
  p_epochs,
  p_dict_size,
  metrics_names,
  score_loss,
  score_accuracy,
  p_train_size_percentage,
  model_optimizer,
  model_loss,
  dictionary_size, number_of_neurons, use_dropout, dropout_fraction, number_of_classes, train_record_count
  ):

    file = "__log.txt"
    f= open(file,"a+")

    if os.path.exists(file) and os.path.getsize(file) > 0:
      f.write(str(datetime.datetime.now()) + "|" + p_input_file_tag_column + "|" + str(p_batch_size) + "|" + str(p_epochs) + "|" + str(p_dict_size) + "|" + str(metrics_names) + "|" + str(score_loss) + "|" + str(score_accuracy) + "|" + str(p_train_size_percentage) + "|" + str(model_optimizer) + "|" + str(model_loss) + "|" + str(dictionary_size) + "|" + str(number_of_neurons) + "|" + str(use_dropout) + "|" + str(dropout_fraction) + "|" + str(number_of_classes) + "|" + str(train_record_count))
      f.write("\r\n")

    else:
      f.write("datetimestamp|tag_column|batch_size|epochs|dict_size|model.metrics_names|test_score_loss|test_score_accuracy|p_train_size_percentage|model_optimizer|model_loss|dictionary_size|number_of_neurons|use_dropout|dropout_fraction|num_of_classes|train_record_count")
      f.write("\r\n")
      
      f.write(str(datetime.datetime.now()) + "|" + p_input_file_tag_column + "|" + str(p_batch_size) + "|" + str(p_epochs) + "|" + str(p_dict_size) + "|" + str(metrics_names) + "|" + str(score_loss) + "|" + str(score_accuracy) + "|" + str(p_train_size_percentage) + "|" + str(model_optimizer) + "|" + str(model_loss) + "|" + str(dictionary_size) + "|" + str(number_of_neurons) + "|" + str(use_dropout) + "|" + str(dropout_fraction) + "|" + str(number_of_classes) + "|" + str(train_record_count))
      f.write("\r\n")

    f.close()

# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(
  cm, classes, title,  model, text_labels, test_texts_matrix,test_tags_encoded_to_categorical
  ):
    #"""
    #    This function prints and plots the confusion matrix.
    #    Normalization can be applied by setting #`normalize=True`.
    #    """
    cmap=plt.cm.Blues,
    ########################################
    ##text_labels = encoder.classes_
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

    ########################################

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



def TestCode():
    
    p_input_file = 'data/TRAINING_FILE_20190509.csv'
        
    data_file = p_input_file
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    print("here 1:...............")
    #################################################
    ## this codes takes the dataset and to prevent the unseen error* takes the first instance of every tag in the test data and adds it to the train data
    #* ValueError: ValueError: y contains previously unseen labels: 'CIGS'
    #* The error comes from this line of code:     test_tags_encoded = encoder.transform(test_tags)

    ddTrain, ddTest = IdentifyNonMatchingValuesBetweenDataSetsv2(df, 0.8, 'category_model_and_manual', 'original_description')

    #print(ddTrain)
    #print(ddTest)

    df = pd.concat([ddTrain, ddTest])



def showModelInfo(p_model_file):

  from keras.models import load_model
  
  model = load_model(p_model_file)
  
  print('\n' + 'model...' + '\n')
  print(model)
  print('metrics_names' + '\n')
  print(model.metrics_names)
  print('\n' + 'optimizer...' + '\n')
  print(model.optimizer)
  #print(model.outputshape)
  print('\n' + 'summary...' + '\n')
  print(model.summary())
  print('\n' + 'config...' + '\n')
  print(model.get_config())
  print('\n' + 'weights...' + '\n')
  print(model.get_weights())
