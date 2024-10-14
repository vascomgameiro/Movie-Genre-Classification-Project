import string

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from rapidfuzz import fuzz



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


def filter_duplicate_descriptions(df: pd.DataFrame, description_col: str, target_col: str) -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows with duplicate descriptions and sorts by description.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    description_col (str): The name of the description column.
    target_col (str): The name of the target column to check for uniqueness.

    Returns:
    pd.DataFrame: The filtered and sorted DataFrame.
    """
    description_target_counts = df.groupby(description_col)[target_col].nunique()
    duplicate_descriptions = description_target_counts[description_target_counts > 1].index
    filtered_df = df[df[description_col].isin(duplicate_descriptions)]
    sorted_filtered_df = filtered_df.sort_values(description_col)
    return sorted_filtered_df


def find_similar_descriptions(df, description_column, cosine_threshold=0.8, jaccard_threshold=0.7):
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 0.0
        return intersection / union

    def tokenize(text):
        return set(text.lower().split())

    descriptions = df[description_column].fillna("")

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
    cosine_sim = cosine_similarity(tfidf_matrix)

    candidate_pairs = []
    for i in range(cosine_sim.shape[0]):
        for j in range(i + 1, cosine_sim.shape[0]):
            if cosine_sim[i, j] >= cosine_threshold:
                candidate_pairs.append((i, j, cosine_sim[i, j]))
    
    final_similar_pairs = []
    for i, j, cos_sim in candidate_pairs:
        set1 = tokenize(df.loc[i, description_column])
        set2 = tokenize(df.loc[j, description_column])
        jac_sim = jaccard_similarity(set1, set2)

        if jac_sim >= jaccard_threshold:
            final_similar_pairs.append((i, j, cos_sim, jac_sim))

    return final_similar_pairs


def print_differences(df, similar_pairs, column_name):
    """
    Prints the differences in the specified column for the given similar pairs.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    similar_pairs (list): List of tuples containing similar pairs and their similarities.
    column_name (str): The name of the column to check for differences.
    """
    print(f"\n Different {column_name.capitalize()}: \n")
    for i, j, cos_sim, jac_sim in similar_pairs:
        value_i = df.loc[i, column_name]
        value_j = df.loc[j, column_name]
        if value_i != value_j:
            print(f"{value_i} ({i}) and {value_j} ({j}) : (Cosine {cos_sim:.4f}, Jaccard {jac_sim:.4f})")


def validate_and_filter_duplicates_fuzzy(df, similar_pairs, columns_to_check, threshold=80):
    """
    Validates potential duplicates by fuzzy matching additional columns and drops the entry with the shorter description.
    Keeps the entry containing the description with more information.

    Parameters:
    df (pd.DataFrame): The original DataFrame.
    similar_pairs (list): List of tuples containing similar description pairs and their similarities.
    columns_to_check (list): List of column names to validate against.
    threshold (int): Fuzzy matching score threshold (0-100).
    
    Returns:
    pd.DataFrame: The DataFrame with duplicates removed.
    """
    # Set to store indices to drop
    index_drop = set()

    for i, j, cos_sim, jac_sim in similar_pairs:
        match = True

        for col in columns_to_check:
            value_i = str(df.loc[i, col])
            value_j = str(df.loc[j, col])
            
            similarity_score = fuzz.ratio(value_i, value_j)
            
            if similarity_score < threshold:
                match = False
                break
        
        if match:
            if len(df.loc[i, 'description']) <= len(df.loc[j, 'description']):
                index_drop.add(i)
            else:
                index_drop.add(j)
    

    df = df.drop(index=index_drop)
    
    return df
