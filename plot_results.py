import pandas as pd
import os
import swifter
import spacy
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize

path = 'C:/Users/Jakob/Documents/RLI Nuclear Energy'

df = pd.read_pickle(os.path.join(path, 'rli-sentence-translation-sentiment-ner.pkl'))
