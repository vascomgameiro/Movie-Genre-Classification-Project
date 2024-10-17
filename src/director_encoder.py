import re

import pandas as pd
from category_encoders import CountEncoder
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def find_similar_descriptions(df, description_column, cosine_threshold=0.6, jaccard_threshold=0.8):
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


def clean_director_name(name: str):
    """
    A director's name is standardized by converting it to lowercase, removing spaces, hyphens, and periods,
    removing any words that have 2 or fewer characters, and removing a trailing comma if present.
    """
    name = name.lower().replace(" ", "").replace("-", "")
    name = re.sub(r"\.", "", name)

    words = re.findall(r"\b\w+\b", name)

    cleaned_words = [word for word in words if len(word) > 2]

    cleaned_name = "".join(cleaned_words)

    if cleaned_name.endswith(","):
        cleaned_name = cleaned_name[:-1]

    return cleaned_name


def fuzzy_director_map(df, similar_pairs, threshold=70):
    """
    Creates a mapping dictionary for director names identified as similar based on fuzzy matching.
    The function will add an entry to the mapping where the value_j will map to value_i if the similarity score is above the threshold.
    This helps in consolidating similar director names across the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the director names to check for similarity.
    similar_pairs (list): A list of tuples with similar director pairs, containing indices and similarity scores.
    threshold (int): The similarity score threshold for fuzzy matching. Only pairs with a score above this threshold will be mapped.

    Returns:
    dict: A dictionary where the keys are director names that should be replaced (value_j) and the values are the standardized director names (value_i).
    """

    director_map = {}

    for i, j, cos_sim, jac_sim in similar_pairs:
        match = True

        value_i = str(df.loc[i, "director"])
        value_j = str(df.loc[j, "director"])

        similarity_score = fuzz.ratio(value_i, value_j)

        if similarity_score < threshold:
            match = False

        if match:
            director_map[value_j] = value_i

    return director_map


def process_director_data(train_df, test_df):
    """
    Concatenates and processes the train and test DataFrames by adding source and original_index columns.
    Splits and cleans the 'director' column.
    """

    # Make copies of the original DataFrames to avoid modifying them
    def add_source_and_index(df, source_name):
        df_copy = df.copy()
        df_copy["original_index"] = df_copy.index
        df_copy["source"] = source_name
        return df_copy

    train_df_copy = add_source_and_index(train_df, "train")
    test_df_copy = add_source_and_index(test_df, "test")

    # Concatenate the DataFrames
    aux_df = pd.concat(
        [
            train_df_copy[["director", "description", "original_index", "source"]],
            test_df_copy[["director", "description", "original_index", "source"]],
        ],
        axis=0,
        ignore_index=True,
    )

    # Remove duplicates, reset index
    aux_df.drop_duplicates(inplace=True)
    aux_df.reset_index(drop=True, inplace=True)

    # Split, explode and strip the 'director' column
    aux_df["director"] = aux_df["director"].str.split(",").explode().reset_index(drop=True)
    aux_df["director"] = aux_df["director"].str.strip()

    # Remove any rows where 'director' is empty
    aux_df = aux_df[aux_df["director"] != ""]

    return aux_df


def get_mapping(aux_df):
    """
    Finds similar descriptions and creates a mapping between the original 'director' column and a new encoded column.
    """
    dictionary_df = aux_df.drop_duplicates(subset=["director", "description"])
    dictionary_df.reset_index(drop=True, inplace=True)
    similar_pairs = find_similar_descriptions(dictionary_df, "description")

    fuzzy_map = fuzzy_director_map(dictionary_df, similar_pairs, threshold=77)
    dictionary_df["director"] = dictionary_df["director"].replace(fuzzy_map)
    dictionary_df.drop_duplicates(subset=["director", "description"], inplace=True)
    dictionary_df.reset_index(drop=True, inplace=True)

    count_encoder = CountEncoder(normalize=True).fit(dictionary_df["director"])
    aux_df["director"] = aux_df["director"].replace(fuzzy_map)
    aux_df["encoded"] = count_encoder.transform(aux_df["director"])
    aux_df.drop_duplicates(subset=["source", "original_index", "encoded"], inplace=True)
    aux_df = aux_df.loc[aux_df.groupby(["source", "original_index"])["encoded"].idxmax()]
    mapping = aux_df.set_index(["source", "original_index"])["encoded"].to_dict()
    return mapping


def encode_directors(df, source_label, mapping):
    temp_df = df.copy()
    temp_df["original_index"] = df.index
    temp_df["source"] = source_label

    df["encoded_director"] = temp_df.apply(lambda row: mapping.get((row["source"], row["original_index"])), axis=1)

    return df


def get_encoding_map(train_df, test_df):
    aux_df = process_director_data(train_df, test_df)
    mapping = get_mapping(aux_df)

    return mapping
