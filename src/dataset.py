import pandas as pd

def get_data(path: str, columns: list[str]) -> pd.DataFrame:

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('\t')
            data.append(fields)
    return pd.DataFrame(data, columns=columns)

import string

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Ensure the required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Initialize stop words
stop_words = set(stopwords.words('english'))

def pos_tagger(nltk_tag):
    """Assign WordNet part of speech tags based on NLTK POS tags."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_tokens(tokens):
    """Lemmatizes tokens in a sentence."""
    pos_tagged = nltk.pos_tag(tokens)
    lemmatized_tokens = [
        nltk.WordNetLemmatizer().lemmatize(token, pos=pos_tagger(tag)) if pos_tagger(tag) else nltk.WordNetLemmatizer().lemmatize(token)
        for token, tag in pos_tagged
    ]
    return lemmatized_tokens

def extract_noun_phrases(sentence):
    """Extract noun phrases from a sentence using TextBlob."""
    blob = TextBlob(sentence)
    return blob.noun_phrases

def preprocess_sentence(sentence):
    """Preprocess a sentence by removing punctuation, stopwords, lemmatizing, and extracting noun phrases."""
    cleaned_sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = word_tokenize(cleaned_sentence)
    noun_phrases = extract_noun_phrases(sentence)
    
    lemmatized_tokens = lemmatize_tokens(tokens)
    preprocessed_tokens = list(set([
        token for token in lemmatized_tokens if token not in stop_words
    ] + noun_phrases))
     
    return preprocessed_tokens