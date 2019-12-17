import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import pickle


def predict_labels(
  p_model_file,
  p_lblencoder_file,
  p_tokenizer_file,
  p_test_data_file,
  p_test_data_file_text_column,
  p_prediction_output_file
  ):
    
    #load the model
    m = load_model(p_model_file)
  
    #load the label encoder
    with open(p_lblencoder_file, 'rb') as handle:
        lblencoder = pickle.load(handle)
    
    # load the tokenizer
    with open(p_tokenizer_file, 'rb') as handle:
        tok = pickle.load(handle)

    #set and take a look at the classes/categories/labels that we trained
    text_labels = lblencoder.classes_
    print (text_labels)

    df = pd.read_csv(p_test_data_file)

    texts = df[p_test_data_file_text_column]

    texts_matrix = tok.texts_to_matrix(texts, mode='count')
    
    df_output = pd.DataFrame(columns=['text', 'label'])
    
    for i in range(0,texts_matrix.shape[0]):
      
      prediction = m.predict(np.array([texts_matrix[i]]))
      predicted_label = text_labels[np.argmax(prediction)]
      
      df_output = df_output.append({'text': texts.iloc[i], 'label':predicted_label}, ignore_index=True)
    

    df_output.to_csv(p_prediction_output_file, index=False, header=True)
    return df_output



def TrainForMultiplelabels(input_file, output_file, file_label):
    #Change the first 3 parameters
    
    #c_text_file = 'data/ExportData.csv' #<--the transaction file you d/l from hsbc that you want to predict on
    #c_prediction_output_file = 'data/hsbc_model_predictions_20190509.csv' #<--this is the file that will be created
    
    c_text_file = input_file
    c_prediction_output_file = output_file
            
    #c_file_date_string = '20190509'
    c_text_file_column = 'original_description'
                
                
    predictions_category = predict_labels(
      'model/multiclass_classifier_hsbc_Categories_' + file_label + '.h5',
      'model/labelencoder_hsbc_Categories_' + file_label + '.pickle',
      'model/tokenizer_hsbc_Categories_' + file_label + '.pickle',
      c_text_file,
      c_text_file_column
      )
                    
    predictions_subcategory = predict_labels(
      'model/multiclass_classifier_hsbc_SubCategories_' + file_label + '.h5',
      'model/labelencoder_hsbc_SubCategories_' + file_label + '.pickle',
      'model/tokenizer_hsbc_SubCategories_' + file_label + '.pickle',
      c_text_file,
      c_text_file_column
      )
                        
    
    df1 = predictions_category
    df2 = predictions_subcategory
    #df3 = predictions_label01
    #df4 = predictions_label02
    #df5 = predictions_provider

    #df_all = pd.concat([df1, df2, df3, df4, df5], axis=1, join='inner')
    df_all = pd.concat([df1, df2], axis=1, join='inner')

    df_all.to_csv(c_prediction_output_file, index=False, header=True)
