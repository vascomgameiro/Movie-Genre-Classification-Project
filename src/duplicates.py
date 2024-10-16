import re
from collections import defaultdict

import pandas as pd
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
            if len(df.loc[i, "description"]) <= len(df.loc[j, "description"]):
                index_drop.add(i)
            else:
                index_drop.add(j)

    df = df.drop(index=index_drop)

    return df


def clean_director_name(name: str):
    """
    A director's name is standardized by converting it to lowercase, removing spaces, hyphens, and periods
    """
    name = name.lower().replace(" ", "").replace("-", "")
    return re.sub(r"\.", "", name)


def create_name_map(df):
    """
    Creates a mapping dictionary where multiple variations of a name will be mapped to a standardized version.
    Keys: Cleaned director names
    Values: Original names

    """
    name_map = defaultdict(set)
    for i, row in df.iterrows():
        director_list = [name.strip() for name in row["director"].split(",")]
        for director in director_list:
            cleaned_name = clean_director_name(director)
            name_map[cleaned_name].add(director)
    return name_map


def map_director_names(df, name_map):
    """
    Maps and replaces director names in the DataFrame with standardized versions using the name map
    """

    # Reverse the name_map for a faster lookup of the standardized names
    reverse_name_map = {}
    for standard_name, original_names in name_map.items():
        for name in original_names:
            reverse_name_map[name] = standard_name

    def apply_mapping(director_names):
        mapped_directors = [reverse_name_map.get(name.strip(), name.strip()) for name in director_names.split(",")]
        return ",".join(mapped_directors)

    df["director"] = df["director"].apply(apply_mapping)
