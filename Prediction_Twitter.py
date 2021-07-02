from gensim.models import VocabTransform
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import gensim
import pickle

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.utils.np_utils import to_categorical

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Switch off chained_assignment warnings
pd.options.mode.chained_assignment = None

COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding=DATASET_ENCODING, names=COLUMNS)
print(df.head())

tweets = df.iloc[:,[5]]
target = df.iloc[:,0]
target[target == 4] = 1

# Clean dataset of missing values
num_missing_desc = df.isnull().sum()[2]
print(f'Number of missing values: {num_missing_desc}')
df = df.dropna()

# Remove @ tags
tweets['text'] = tweets['text'].map(lambda x: re.sub("@\S+", ' ', x))

# Convert all tweets to lower case
tweets['text'] = tweets['text'].map(lambda x: x.lower())

# Remove numbers
tweets['text'] = tweets['text'].map(lambda x: re.sub(r'\d+', ' ', x))

# Remove links
links_re = "https?:\S+|http?:\S|[^A-Za-z0-9]+"
tweets['text'] = tweets['text'].map(lambda x: re.sub(links_re, ' ', x))

# Remove Punctuations
tweets['text'] = tweets['text'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))

# Remove white spaces
tweets['text'] = tweets['text'].map(lambda x: x.strip())

# Tokenize text
tweets['text'] = tweets['text'].map(lambda x: word_tokenize(x))

# Remove non alphabetic tokens
tweets['text'] = tweets['text'].map(lambda x: [word for word in x if word.isalpha()])

# Remove stop words
stop_words = set(stopwords.words('english'))
tweets['text'] = tweets['text'].map(lambda x: [word for word in x if not word in stop_words])

# Word lemmantization
lem = WordNetLemmatizer()
tweets['text'] = tweets['text'].map(lambda x: [lem.lemmatize(word, 'v') for word in x])

# Turn list back into string
tweets['text'] = tweets['text'].map(lambda x: ' '.join(x))

X_train, X_test, y_train, y_test = train_test_split(tweets, target, test_size=0.2, random_state=42)
print(f'Size of training set: {len(X_train)}')
print(f'Size of testing set: {len(X_test)}')

W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

documents = [info.split() for info in X_train.text]
w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT, 
                                            workers=8)

w2v_model.build_vocab(documents)
words = w2v_model.wv.key_to_index
vocab_size = len(words)
print(f'Vocab_size: {vocab_size}')

# Train W2V model
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

# Test model
print(w2v_model.wv.most_similar("hate"))

MAX_SEQ_LEN = 300
EMBEDDING_DIM = 300

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train.text)
word_index = tokenizer.word_index
vocab_size = len(word_index)
print(f'Found {vocab_size} unique tokens')

# Convert data to padded sequences
X_train_padded = tokenizer.texts_to_sequences(X_train.text)
X_train_padded = pad_sequences(X_train_padded, maxlen=MAX_SEQ_LEN)
print(f'Shape of data tensor: {X_train_padded.shape}')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

embedding_matrix = np.zeros((vocab_size + 1, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

