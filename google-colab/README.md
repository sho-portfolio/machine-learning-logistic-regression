# MachineLearning MultiClassClassifier (Keras)

## Description:  multiclass classification of text inputs using Keras and no math in site!
### code to train, evaluate, productionize and use a model to predict in ~35 lines of code 
##### you can run the run the code directly here (you may need to sign in) https://repl.it/@smtest/KerasMultiClassClassificationSimple

#### Four main sections:               
* [1] prepare the data
* [2] train and evaluate model               
* [3] productionize model (save the model and associated files)               
* [4] use the saved model to make predictions

## Quickly see this code in action!
if you'd like to see this code in action go here and press Run!
https://repl.it/@smtest/KerasMultiClassClassificationSimple

There's some real data (Stack Overflow posts) in the realData folder that you can use to put the model through it's paces but you may need to run this on a more powerful environment (your Mac or PC will easily suffice)


#### [1] prepare the data
###### nothing magical here, your dataset is comprised of words but a model needs numbers as it's input.  This code does that conversion!

```python
# load the dataset
df = pd.read_csv('dataTrain.txt')

# define the input and label columns
texts = df['text']
labels = df['label']

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
```

#### [2] train and evaluate model 
###### this code simply defines the model, then builds it (fit) then checks to see how accurate it is by running it over the validation data (X_test_enc and y_test_enc)

```python
#specify the model parmeters
epochs=40
number_of_neurons = 500
batch_size = 128
dropout_fraction = 0.5
vocab_size = len(tok.word_index) + 1

#define the model
model = Sequential()
model.add(Dense(number_of_neurons, input_shape=(vocab_size,), activation='relu'))
model.add(Dropout(dropout_fraction, seed=None))
model.add(Dense(num_classes)) 
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
print(model.summary())

#fit the model
model.fit(X_train_enc, y_train_enc_categorical, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

#evaluate the accuracy of the trained model
score = model.evaluate(X_test_enc, y_test_enc_categorical, batch_size=batch_size, verbose=1)
print("score: ", score)
```

### [3] productionize model (save the model and associated files)  
```python
#Save the model (so that it can be used outside of this notebook)
model.save('model.h5')

#Save/Pickle the tokenizer
with open('tokenizer.pickle', 'wb') as handle: pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Save/Pickle the Label Encoder
with open('labelencoder.pickle', 'wb') as handle: pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

### [4] use the saved model to make predictions
###### this bit uses the saved model and runs it against a set of data previosuly unseen by the model, you can then check to see how well it predicts!
```python
#load the model
model = load_model('model.h5')

#load the label encoder
with open('labelencoder.pickle', 'rb') as handle: le = pickle.load(handle)

# load the tokenizer
with open('tokenizer.pickle', 'rb') as handle: tok = pickle.load(handle)


#create a dataframe from the test datafile
df = pd.read_csv('dataTest.txt')

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
```



# Resources & Next Steps

### Resources that were instrumental in helping with the creation of this code - a huge thank you to the authors!
* [tokenizer] https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/#targetText=Keras%20provides%20the%20Tokenizer%20class,or%20integer%20encoded%20text%20documents
* [labelencoder] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
* https://datascience.stackexchange.com/questions/46124/what-do-compile-fit-and-predict-do-in-keras-sequential-models
* https://cloud.google.com/blog/products/gcp/intro-to-text-classification-with-keras-automatically-tagging-stack-overflow-posts
* https://github.com/tensorflow/workshops/blob/master/extras/keras-bag-of-words/keras-bow-model.ipynb
* https://debuggercafe.com/text-preprocessing-with-keras-4-simple-ways/
* https://www.kaggle.com/jacklinggu/keras-mlp-cnn-test-for-text-classification/data
* https://github.com/keras-team/keras/tree/master/keras
* https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance
* https://github.com/keras-team/keras/blob/master/keras/metrics.py
* https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
* https://machinelearningmastery.com/what-are-word-embeddings/
* https://github.com/scikit-learn-contrib/categorical-encoding/issues/26
* https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
* https://machinelearningmastery.com/difference-test-validation-datasets/
* https://www.kaggle.com/jacklinggu/keras-mlp-cnn-test-for-text-classification/data
* https://github.com/keras-team/keras/tree/master/keras
* https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance
* https://github.com/keras-team/keras/blob/master/keras/metrics.py
* https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
* https://machinelearningmastery.com/what-are-word-embeddings/
* https://github.com/scikit-learn-contrib/categorical-encoding/issues/26
* http://faroit.com/keras-docs/1.2.2/preprocessing/text/
* https://keras.io/preprocessing/text/

### Read later list
* https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
* https://cloud.google.com/blog/products/gcp/intro-to-text-classification-with-keras-automatically-tagging-stack-overflow-posts
* https://github.com/keras-team/keras/issues/7985
* https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
* https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
* [JSON/YAML] https://machinelearningmastery.com/save-load-keras-deep-learning-models/



### Next Steps (#todo)
* a) add code to displaying confusion matrix visual
* b) modularize code and bring all variables/parameteres to the top (not done here to make the code easier to read and follow)
* c) rename variables to show the difference between train, validate and test data and write function to split 1 dataset into these three components
* d) allow for option to select optimizer (Adam vs SGD for example) and to tune the instantiator variables for each optimizer
* e) code to allow for multiple inputs (text) - currently only one dimension (column) is used as input
* f) code to write all parameters/variables and model accuracy/loss and execution time values to log file
* g) save model as json or yaml (see resource list TAG [JSON/YAML])
* h) write a data cleaning module to remove records with no labels or texts and empty lines (dont modify original data just create a clean dataframe at runtime)
* i) write code to help user decide the vocab_size in case it's just too big such as the stack-overflow dataset which has 135,000+.  we can use vocab_size = len(tok.word_index) + 1 if we have a powerful processor but it slows down the model training by a factor of upto 100s without much or any improvement in accuracy
