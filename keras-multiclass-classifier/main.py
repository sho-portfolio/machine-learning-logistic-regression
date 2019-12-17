#import _helper as helper
#helper.showModelInfo("keras_model_multiclass_classifier_Label.h5")
#exit()

import _train as train
import _predict as predict

## Specify the training file information
## Simple
training_data_file = 'data/simple/trainData.txt'
training_data_file_text_col = 'Text'
training_data_file_label_col = 'Label'
## Hsbc
#training_data_file = 'data/hsbc/trainData.txt'
#training_data_file_text_col = 'Original Description'
#training_data_file_label_col = 'category_model_and_manual'

## Specify the prediction run parameteres
## Simple
test_data_file = 'data/simple/testData.txt'
prediction_output_file = 'data/simple/predictedData.txt'
test_data_file_text_column = 'Text'
## Hsbc
#test_data_file = 'data/hsbc/trainData.txt'
#prediction_output_file = 'data/hsbc/predictedData.txt'
#test_data_file_text_column = 'Original Description'


#specify the model parmeters
model_optimizer = 'Adam'
epochs=1
number_of_neurons = 1000
batch_size = 32
dict_size=5000 #set to 5000: this is usually a lot lower if it's just for real english words as there are not that many words in the english language
use_dropout = False
dropout_fraction = 0.5
train_size_percentage = 0.8

#specify other misc. parameters
plot_confusion_matrix = False
#output_file = 'data/model_predictions.txt'

"""
print("****train model****")
train.f_train_model(
    p_model_optimizer = model_optimizer,
    p_training_data_file = training_data_file,
    p_training_data_file_text_column = training_data_file_text_col,
    p_training_data_file_label_column = training_data_file_label_col,
    p_model_file = 'keras_model_multiclass_classifier_' + training_data_file_label_col + '.h5',
    p_tokenizer_file = 'tokenizer_' + training_data_file_label_col + '.pickle',
    p_labelencoder_file = 'labelencoder_' + training_data_file_label_col + '.pickle',
    p_batch_size = batch_size,
    p_epochs=epochs,
    p_dict_size=dict_size,
    p_train_size_percentage = train_size_percentage,
    p_plot_confusion_matrix = plot_confusion_matrix,
    p_number_of_neurons = number_of_neurons,
    p_use_dropout = use_dropout,
    p_dropout_fraction = dropout_fraction
)
"""

print("****test/run model****")
df_predicted = predict.predict_labels(
  p_model_file = 'keras_model_multiclass_classifier_' + training_data_file_label_col + '.h5',
  p_lblencoder_file = 'labelencoder_' + training_data_file_label_col + '.pickle',
  p_tokenizer_file = 'tokenizer_' + training_data_file_label_col + '.pickle',
  p_test_data_file = test_data_file,
  p_test_data_file_text_column = test_data_file_text_column,
  p_prediction_output_file = prediction_output_file
  )
print(df_predicted)

print("execution complete...")
