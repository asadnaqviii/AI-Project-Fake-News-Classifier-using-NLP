import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re
from nltk.stem import PorterStemmer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm


# Load the dataset
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

# Combine the datasets and shuffle
fake['label'] = 0
real['label'] = 1
data = pd.concat([fake, real]).sample(frac=1, random_state=42)

# Combine text and title for both fake and real news
data['total'] = data['title'] + ' ' + data['text']

# Preprocessing
stemmer = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(text)

data['total'] = data['total'].apply(preprocess_text)

# Split the dataset into training, validation, and test sets
train_ratio = 0.64
validation_ratio = 0.16
test_ratio = 0.20

# train is now 64% of the entire data set
# the _junk suffix means that we drop that variable completely
train, test = train_test_split(data, test_size=1 - train_ratio, random_state=42)

# test is now 20% of the initial data set
# validation is now 16% of the initial data set
val, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42) 

# Tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['total'])
train_sequences = tokenizer.texts_to_sequences(train['total'])
val_sequences = tokenizer.texts_to_sequences(val['total'])
test_sequences = tokenizer.texts_to_sequences(test['total'])

max_len = max([len(x) for x in train_sequences])
train_sequences = pad_sequences(train_sequences, maxlen=max_len)
val_sequences = pad_sequences(val_sequences, maxlen=max_len)
test_sequences = pad_sequences(test_sequences, maxlen=max_len)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define model checkpoint
model_checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, save_freq='epoch')

model.fit(train_sequences, train['label'], validation_data=(val_sequences, val['label']), epochs=2, callbacks=[model_checkpoint])

# Predict classes
probabilities = model.predict(test_sequences)
predictions = (probabilities > 0.5).astype(int)


# Calculate metrics
accuracy = accuracy_score(test['label'], predictions)
precision = precision_score(test['label'], predictions)
recall = recall_score(test['label'], predictions)
f1 = f1_score(test['label'], predictions)

print('Accuracy: ', accuracy)
print('Precision: ', precision)

print('Recall: ', recall)
print('F1 score: ', f1)

def predict_fake_news(text):
    # Preprocess the text
    text = preprocess_text(text)

    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_len)

    prediction = model.predict_classes(sequence)
    
    return 'Real' if prediction[0][0] == 1 else 'Fake'

# Apply the preprocess function to each text in the 'total' column with a progress bar
tqdm.pandas(desc="Preprocessing texts")
data['total'] = data['total'].progress_apply(preprocess_text)

# Fit the tokenizer on the training texts with a progress bar
print("Fitting tokenizer")
for text in tqdm(train['total']):
    tokenizer.fit_on_texts(text)

# Convert texts to sequences with a progress bar
print("Converting texts to sequences")
train_sequences = [tokenizer.texts_to_sequences(text) for text in tqdm(train['total'])]
val_sequences = [tokenizer.texts_to_sequences(text) for text in tqdm(val['total'])]
test_sequences = [tokenizer.texts_to_sequences(text) for text in tqdm(test['total'])]


#The performance of a model on a particular iteration (or epoch) depends on both its loss and its accuracy. Lower loss and higher accuracy are desirable.

#In the first iteration, the model achieved a validation loss of 0.0535 and a validation accuracy of 0.9823. In the second iteration, the model achieved a validation loss of 0.3292 and a validation accuracy of 0.8769.

#Given these numbers, the first iteration performed better. Although the model's training loss and accuracy improved in the second iteration (lower loss, higher accuracy), the validation loss increased significantly and the validation accuracy decreased. This discrepancy could suggest that the model is overfitting, meaning it's learning the training data too well and is not generalizing effectively to unseen data (validation set).