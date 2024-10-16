from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def plot_genre_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(y="genre", data=df, order=df["genre"].value_counts().index)
    plt.title("Distribution of Movie Genres")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.show()


def plot_top_directors(df):
    plt.figure(figsize=(12, 6))
    top_directors = df["director"].value_counts().head(10)
    sns.barplot(x=top_directors.values, y=top_directors.index)
    plt.title("Top 10 Most Popular Directors")
    plt.xlabel("Number of Movies")
    plt.ylabel("Director")
    plt.show()


def plot_genre_vs_director(df):
    top_10_directors = df["director"].value_counts().head(10).index
    filtered_df = df[df["director"].isin(top_10_directors)]
    plt.figure(figsize=(14, 7))
    sns.countplot(y="director", hue="genre", data=filtered_df)
    plt.title("Top 10 Directors by Genre")
    plt.xlabel("Number of Movies")
    plt.ylabel("Director")
    plt.legend(loc="upper right")
    plt.show()


def plot_movie_data(df):
    sns.set_theme(style="whitegrid")
    plot_genre_distribution(df)
    plot_top_directors(df)
    plot_genre_vs_director(df)


def plot_stopword_frequency(df: pd.DataFrame, column_name: str):
    stop_words = set(stopwords.words("english"))
    stopwords_in_titles = (
        df[column_name].str.lower().str.split().apply(lambda x: [word for word in x if word in stop_words])
    )
    all_stopwords = [word for sublist in stopwords_in_titles for word in sublist]
    stopword_counts = Counter(all_stopwords)

    stopword_df = pd.DataFrame(stopword_counts.items(), columns=["Stopword", "Frequency"]).sort_values(
        by="Frequency", ascending=False
    )
    plt.figure(figsize=(20, 6))
    plt.bar(stopword_df["Stopword"], stopword_df["Frequency"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(stats_df):
    """
    Plot the correlation matrix of the text statistics.

    Parameters:
    stats_df (pd.DataFrame): DataFrame containing the text statistics.
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = stats_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Text Statistics")
    plt.show()


def plot_pca(data, genre_labels):
    """
    Plot PCA to reduce the text statistics to two dimensions and plot the results, colored by genre labels.

    Parameters:
    data (ndarray): Scaled data for PCA input.
    genre_labels (pd.Series): The genre labels for each document.

    Returns:
    pd.DataFrame: Transformed data in 2D.
    PCA: The PCA object for further analysis.
    """
    # Apply PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    # Convert PCA result to DataFrame
    pca_df = pd.DataFrame(pca_data, columns=["PCA1", "PCA2"])
    pca_df["Genre"] = genre_labels.values

    # Plot PCA result, colored by genre
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="PCA1", y="PCA2", hue="Genre", palette="Set2", data=pca_df)
    plt.title("PCA Projection of Text Statistics (2D) - Colored by Genre")
    plt.show()

    return pca_df, pca


def plot_pca_tfidf(df, text_column, genre_column):
    """
    Perform TF-IDF vectorization of the text column, apply PCA for dimensionality reduction to 2D,
    and plot the results, coloured by genre labels.

    Parameters:
    df (pd.DataFrame): DataFrame containing text and genre columns.
    text_column (str): Name of the column with text data.
    genre_column (str): Name of the column with genre labels.

    Returns:
    pd.DataFrame: Transformed data in 2D after PCA.
    PCA: The PCA object for further analysis.
    """
    # Step 1: TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features based on dataset size
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column].values.astype("U"))

    # Step 2: Standardize the TF-IDF data (optional but recommended for PCA)
    scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse data
    scaled_data = scaler.fit_transform(tfidf_matrix.toarray())

    # Step 3: Apply PCA to reduce to 2 components
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Step 4: Create a DataFrame for the PCA result
    pca_df = pd.DataFrame(pca_data, columns=["PCA1", "PCA2"])
    pca_df["Genre"] = df[genre_column].values

    # Step 5: Plot the PCA result, colored by genre
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="PCA1", y="PCA2", hue="Genre", palette="Set2", data=pca_df)
    plt.title("PCA Projection of TF-IDF Features (2D) - Colored by Genre")
    plt.show()

    return pca_df, pca


def get_text_statistics(df, text_col):
    """
    Calculate descriptive statistics for a text column, including token counts, sentence counts, document lengths, etc.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    text_col (str): The name of the text column.

    Returns:
    pd.DataFrame: A DataFrame containing statistics for each document.
    """
    stats = {
        "num_tokens_per_doc": [],
        "num_sentences_per_doc": [],
        "avg_sentence_length_per_doc": [],
        "avg_doc_length": [],
        "num_chars_per_doc": [],
        "sentence_length_variance": [],
    }

    for text in df[text_col]:
        # Tokenize by words and sentences
        tokens = nltk.word_tokenize(text)
        sentences = nltk.sent_tokenize(text)

        # Document level stats
        stats["num_tokens_per_doc"].append(len(tokens))
        stats["num_sentences_per_doc"].append(len(sentences))
        stats["avg_sentence_length_per_doc"].append(len(tokens) / len(sentences) if len(sentences) > 0 else 0)
        stats["avg_doc_length"].append(len(tokens))  # Total token count for document length
        stats["num_chars_per_doc"].append(len(text))  # Total character count

        # Sentence length variance
        sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
        variance = np.var(sentence_lengths) if sentence_lengths else 0
        stats["sentence_length_variance"].append(variance)

    return pd.DataFrame(stats)


def plot_histograms(stats):
    """
    Plot histograms for text statistics like token count distribution, sentence count distribution, etc.

    Parameters:
    stats (dict): A dictionary containing the text statistics (from get_text_statistics).
    """
    # Plot token count distribution
    plt.figure(figsize=(12, 5))
    sns.histplot(stats["num_tokens_per_doc"], kde=True, bins=30)
    plt.title("Distribution of Token Count per Document")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.show()

    # Plot sentence count distribution
    plt.figure(figsize=(12, 5))
    sns.histplot(stats["num_sentences_per_doc"], kde=True, bins=30)
    plt.title("Distribution of Sentence Count per Document")
    plt.xlabel("Number of Sentences")
    plt.ylabel("Frequency")
    plt.show()

    # Plot average sentence length per document
    plt.figure(figsize=(12, 5))
    sns.histplot(stats["avg_sentence_length_per_doc"], kde=True, bins=30)
    plt.title("Average Sentence Length per Document")
    plt.xlabel("Average Sentence Length")
    plt.ylabel("Frequency")
    plt.show()

    # Plot average document length (number of tokens)
    plt.figure(figsize=(12, 5))
    sns.histplot(stats["avg_doc_length"], kde=True, bins=30)
    plt.title("Average Document Length (Token Count)")
    plt.xlabel("Document Length (Number of Tokens)")
    plt.ylabel("Frequency")
    plt.show()


def plot_boxplots(stats):
    """
    Plot boxplots for text statistics like token count, sentence count, etc.

    Parameters:
    stats (dict): A dictionary containing the text statistics (from get_text_statistics).
    """
    # Boxplot for token count per document
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=stats["num_tokens_per_doc"])
    plt.title("Boxplot of Token Count per Document")
    plt.xlabel("Number of Tokens")
    plt.show()

    # Boxplot for sentence count per document
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=stats["num_sentences_per_doc"])
    plt.title("Boxplot of Sentence Count per Document")
    plt.xlabel("Number of Sentences")
    plt.show()

    # Boxplot for average sentence length per document
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=stats["avg_sentence_length_per_doc"])
    plt.title("Boxplot of Average Sentence Length per Document")
    plt.xlabel("Average Sentence Length")
    plt.show()

    # Boxplot for document length (number of tokens)
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=stats["avg_doc_length"])
    plt.title("Boxplot of Document Length (Token Count)")
    plt.xlabel("Document Length (Number of Tokens)")
    plt.show()
