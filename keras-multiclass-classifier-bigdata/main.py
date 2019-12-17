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



def getvocabsize(X_train):
  tok = Tokenizer()
  tok.fit_on_texts(X_train)
  return(len(tok.word_index) + 1)

## train a model

# load the dataset
df = pd.read_csv('dataTrainSample.csv')

texts = df['post']
labels = df['tags']

# Split data into train and test (80% used for training, 20% for )
train_size = int(len(df) * 0.8)
X_train = texts[:train_size]
y_train = labels[:train_size]
X_test = texts[train_size:]
y_test = labels[train_size:]

vocab_size = min(getvocabsize(X_train), 1000)

# prepare (encode) input data (using tokenizer)
tok = Tokenizer(vocab_size)
tok.fit_on_texts(X_train)
X_train_enc = tok.texts_to_matrix(X_train, mode='count')
X_test_enc = tok.texts_to_matrix(X_test, mode='count')

# prepare (encode) target data
num_classes = y_train.nunique()
le = LabelEncoder()
le.fit_transform(y_train)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test) #test labels transformed but not fit
y_train_enc_categorical = utils.to_categorical(y_train_enc, num_classes)
y_test_enc_categorical = utils.to_categorical(y_test_enc, num_classes)

#specify the model parmeters
epochs=15
number_of_neurons = 1000
batch_size = 2000
dropout_fraction = 0.5



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

#Save the model (so that it can be used outside of this notebook)
model.save('model.h5')

#Save/Pickle the tokenizer
with open('tokenizer.pickle', 'wb') as handle: pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Save/Pickle the Label Encoder
with open('labelencoder.pickle', 'wb') as handle: pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)


## Use the saved model to make predictions

#load the model
model = load_model('model.h5')

#load the label encoder
with open('labelencoder.pickle', 'rb') as handle: le = pickle.load(handle)

# load the tokenizer
with open('tokenizer.pickle', 'rb') as handle: tok = pickle.load(handle)


#create a dataframe from the test datafile
df = pd.read_csv('dataTestSample.csv')

#create the output dataframe
df_output = pd.DataFrame(columns=['post', 'tags'])

#encode the text values we want to predict on using the loaded tokenizer pickle file
X_test = df['post']
X_test_enc = tok.texts_to_matrix(X_test, mode='count')

text_labels = le.classes_
for i in range(0,X_test_enc.shape[0]):
  prediction = model.predict(np.array([X_test_enc[i]]))
  predicted_label = text_labels[np.argmax(prediction)]
  #print("input: ", X_test[i])
  #print("prediction: ", prediction)
  #print("predicted_label: ", predicted_label)
  df_output = df_output.append({'post': X_test.iloc[i], 'tags':predicted_label}, ignore_index=True)

print (df_output)
df_output.to_csv('dataTestSamplePredictions.csv', index=False, header=True)

