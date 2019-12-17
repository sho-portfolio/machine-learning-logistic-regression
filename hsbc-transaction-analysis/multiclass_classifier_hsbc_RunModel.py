#Author:  Sho Munshi
#Desc:    First ever production model use! The concepts here are very poorly documented
#         and difficult to find/understand to new comers
#         Pass in hsbc transaction data (statement transaction(s)
#         to predict the category labels of statement transactions
#Date:    2019-02-28 (Completed)

##########################################################################################################

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra

from keras.models import load_model
from keras.preprocessing.text import Tokenizer

import pickle

##########################################################################################################

# Generate a prediction for many inputs (then output them to csv)

#print (df_output)
#df_output.to_csv('../_data/hsbc_model_predictions_categories_20190317.csv', index=False, header=True)

def predict_labels(
    p_model_file, 
    p_lblencoder_file, 
    p_tokenizer_file, 
    p_text_file, 
    p_text_file_column
):

    #load the model
    m = load_model(p_model_file)

    #load the label encoder
    with open(p_lblencoder_file, 'rb') as handle:
        lblencoder = pickle.load(handle)

    # load the tokenizer
    with open(p_tokenizer_file, 'rb') as handle:
        tok = pickle.load(handle)
    
    #set and take a look at the classes/categories/labels
    text_labels = lblencoder.classes_ 
    print (text_labels)
    
    data = p_text_file
    df = pd.read_csv(data)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    texts = df[p_text_file_column]

    texts_matrix = tok.texts_to_matrix(texts, mode='count')

    df_output = pd.DataFrame(columns=['text', 'tag'])

    for i in range(0,texts_matrix.shape[0]):
    
        prediction = m.predict(np.array([texts_matrix[i]]))
        predicted_label = text_labels[np.argmax(prediction)]
    
        df_output = df_output.append({'text': texts.iloc[i], 'tag':predicted_label}, ignore_index=True)
    
    return df_output

##########################################################################################################

#Change the first 3 parameters

c_text_file = '../_data/hsbc/20190509_export.csv' #<--the transaction file you d/l from hsbc that you want to predict on
c_prediction_output_file = '../_data/hsbc/hsbc_model_predictions_20190509.csv' #<--this is the file that will be created
c_file_date_string = '20190509'
c_text_file_column = 'original_description'


predictions_category = predict_labels(
    '../_model/multiclass_classifier_hsbc_Categories_' + c_file_date_string + '.h5',
    '../_model/labelencoder_hsbc_Categories_' + c_file_date_string + '.pickle',
    '../_model/tokenizer_hsbc_Categories_' + c_file_date_string + '.pickle',
    c_text_file,
    c_text_file_column
)

predictions_subcategory = predict_labels(
    '../_model/multiclass_classifier_hsbc_SubCategories_' + c_file_date_string + '.h5',
    '../_model/labelencoder_hsbc_SubCategories_' + c_file_date_string + '.pickle',
    '../_model/tokenizer_hsbc_SubCategories_' + c_file_date_string + '.pickle',
    c_text_file,
    c_text_file_column
)

predictions_tag01 = predict_labels(
    '../_model/multiclass_classifier_hsbc_Tag01_' + c_file_date_string + '.h5',
    '../_model/labelencoder_hsbc_Tag01_' + c_file_date_string + '.pickle',
    '../_model/tokenizer_hsbc_Tag01_' + c_file_date_string + '.pickle',
    c_text_file,
    c_text_file_column
)

predictions_tag02 = predict_labels(
    '../_model/multiclass_classifier_hsbc_Tag02_' + c_file_date_string + '.h5',
    '../_model/labelencoder_hsbc_Tag02_' + c_file_date_string + '.pickle',
    '../_model/tokenizer_hsbc_Tag02_' + c_file_date_string + '.pickle',
    c_text_file,
    c_text_file_column
)

predictions_provider = predict_labels(
    '../_model/multiclass_classifier_hsbc_Provider_' + c_file_date_string + '.h5',
    '../_model/labelencoder_hsbc_Provider_' + c_file_date_string + '.pickle',
    '../_model/tokenizer_hsbc_Provider_' + c_file_date_string + '.pickle',
    c_text_file,
    c_text_file_column
)

df1 = predictions_category
df2 = predictions_subcategory
df3 = predictions_tag01
df4 = predictions_tag02
df5 = predictions_provider


df_all = pd.concat([df1, df2, df3, df4, df5], axis=1, join='inner')

df_all.to_csv(c_prediction_output_file, index=False, header=True)

##########################################################################################################
#############################EVERYTHING BELOW HERE IS FOR DEBUGGING PURPOSES#############################
##########################################################################################################

df_all

##########################################################################################################

#load the model
m = load_model('../_model/multiclass_classifier_hsbc_Categories_20190317.h5')

#load the label encoder
with open('../_model/labelencoder_hsbc_Categories_20190317.pickle', 'rb') as handle:
    lblencoder = pickle.load(handle)

# load the tokenizer
with open('../_model/tokenizer_hsbc_Categories_20190317.pickle', 'rb') as handle:
    tok = pickle.load(handle)
    
#set and take a look at the classes/categories/labels
text_labels = lblencoder.classes_ 
print (text_labels)   

##########################################################################################################

# Generate a prediction for a single input

txtinput = 'LYFT   *RIDE FRI 3PM, xxxxxx0278   CA, REF NO 5xx2950NAJJ23BS2R, TRAN CD 253 SIC CD 4121'
txtinput_series = pd.Series(txtinput)
txtinput_series_matrix = tok.texts_to_matrix(txtinput_series, mode='count')

prediction = m.predict(np.array(txtinput_series_matrix))
predicted_label = text_labels[np.argmax(prediction)]

print("Input: " + txtinput)
print("Predicted Label: " + predicted_label)

##########################################################################################################

# Generate a prediction for many inputs (then output them to csv)
data = '../_data/hsbc_data_to_label.csv'
df = pd.read_csv(data)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

texts = df.original_description
texts_matrix = tok.texts_to_matrix(texts, mode='count')

df_output = pd.DataFrame(columns=['text', 'tag'])

for i in range(0,texts_matrix.shape[0]):
    
    prediction = m.predict(np.array([texts_matrix[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    
    df_output = df_output.append({'text': texts.iloc[i], 'tag':predicted_label}, ignore_index=True)
    
print (df_output)
df_output.to_csv('../_data/hsbc_model_predictions_categories_20190317.csv', index=False, header=True)
