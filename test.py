import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re
from nltk.stem import PorterStemmer
from keras.models import load_model
from tqdm import tqdm
import pickle

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
train, test = train_test_split(data, test_size=1 - train_ratio, random_state=42)
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

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the model
model = load_model('model.h5')

# Predict probabilities
probabilities = model.predict(test_sequences)

# Convert probabilities to classes
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

    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_len)

    # Predict probabilities for the text
    probabilities = model.predict(sequence)

    # Convert probabilities to classes
    prediction = (probabilities > 0.5).astype(int)
    
    return 'Real' if prediction[0][0] == 1 else 'Fake'
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re
from nltk.stem import PorterStemmer
from keras.models import load_model
from tqdm import tqdm
import pickle

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
train, test = train_test_split(data, test_size=1 - train_ratio, random_state=42)
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

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the model
model = load_model('model.h5')

# Predict probabilities
probabilities = model.predict(test_sequences)

# Convert probabilities to classes
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

# def predict_fake_news(text):
#     # Preprocess the text
#     text = preprocess_text(text)

#     sequence = tokenizer.texts_to_sequences([text])
#     sequence = pad_sequences(sequence, maxlen=max_len)

#     # Predict probabilities for the text
#     probabilities = model.predict(sequence)

#     # Convert probabilities to classes
#     prediction = (probabilities > 0.5).astype(int)
    
#     return 'Real' if prediction[0][0] == 1 else 'Fake'

