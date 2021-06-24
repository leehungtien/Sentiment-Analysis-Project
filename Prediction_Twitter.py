import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding=DATASET_ENCODING, names=COLUMNS)
print(df.head())

X = df.iloc[:,[5]]
Y = df.iloc[:,0]
Y[Y == 4] = 1

# Clean dataset of missing values
num_missing_desc = df.isnull().sum()[2]
print(f'Number of missing values: {num_missing_desc}')
df.dropna()


