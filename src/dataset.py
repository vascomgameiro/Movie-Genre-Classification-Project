import string

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from textblob import TextBlob


def read_data(path: str, columns: list[str]) -> pd.DataFrame:
    """Read data from a file into a Pandas DataFrame.

    Args:
        path (str): The path to the file to read.
        columns (list[str]): The column names.

    Returns:
        pd.DataFrame: The DataFrame containing the data.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split("\t")
            data.append(fields)

    return pd.DataFrame(data, columns=columns)


stop_words = set(stopwords.words("english"))


def pos_tagger(nltk_tag: str) -> str:
    """Map NLTK POS tags to WordNet POS tags."""
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_tag.startswith("R"):
        return wordnet.ADV
    return ""


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize tokens based on their POS tags."""
    lemmatizer = nltk.WordNetLemmatizer()
    pos_tagged = nltk.pos_tag(tokens)
    return [
        lemmatizer.lemmatize(token, pos_tagger(tag)) if pos_tagger(tag) else lemmatizer.lemmatize(token)
        for token, tag in pos_tagged
    ]


def extract_noun_phrases(sentence: str) -> list[str]:
    """Extract noun phrases from a sentence."""
    return [phrase for phrase in getattr(TextBlob(sentence), "noun_phrases") if len(phrase.split()) > 1]


def preprocess_sentence(sentence: str) -> str:
    """Preprocess the sentence by tokenizing, lemmatizing, removing stopwords, and joining noun phrases."""
    cleaned_sentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower()
    tokens = word_tokenize(cleaned_sentence)
    noun_phrases = extract_noun_phrases(sentence)
    noun_phrases_joined = ["".join(phrase.split()) for phrase in noun_phrases]
    lemmatized_tokens = lemmatize_tokens(tokens)
    filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]
    combined_tokens = list(set(filtered_tokens + noun_phrases_joined))
    return " ".join(combined_tokens)
