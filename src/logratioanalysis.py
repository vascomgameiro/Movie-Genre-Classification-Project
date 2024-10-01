import math
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd


class LogRatioAnalysis:
    def __init__(self, df, column, target_column, tokenizer=None):
        """
        Initialize with a DataFrame, the column to analyze, and the target column.
        """
        self.df = df
        self.column = column
        self.target_column = target_column
        self.tokenizer = tokenizer if tokenizer else nltk.word_tokenize  # Default to nltk word tokenizer

    def _calculate_word_frequencies(self, corpus):
        """
        Returns a Counter of word frequencies in a given list of text documents.
        """
        word_counts = Counter()
        for document in corpus:
            if isinstance(document, str):
                tokens = self.tokenizer(document)  # Use the tokenizer to tokenize the document
                word_counts.update(tokens)
        return word_counts

    def _get_corpora(self, target_class):
        """
        Splits the data into two corpora based on the target class.
        """
        corpus_1 = self.df[self.df[self.target_column] == target_class][self.column].tolist()
        corpus_2 = self.df[self.df[self.target_column] != target_class][self.column].tolist()
        return corpus_1, corpus_2

    def _log_ratio_analysis(self, freqs_1, freqs_2):
        """
        Performs log ratio analysis and returns the log ratio for each word.
        """
        log_ratios = {}
        total_1 = sum(freqs_1.values())
        total_2 = sum(freqs_2.values())

        for word in set(freqs_1.keys()).union(freqs_2.keys()):
            # Add smoothing to avoid division by zero
            p1 = (freqs_1.get(word, 0) + 1) / total_1
            p2 = (freqs_2.get(word, 0) + 1) / total_2
            log_ratios[word] = math.log2(p1 / p2)

        return log_ratios

    def get_log_ratios(self, target_class):
        """
        Main method to get log ratios. Takes the target class as input,
        processes the DataFrame, and computes the log ratio of word frequencies.
        """
        # Step 1: Get the two corpora based on the target class
        corpus_1, corpus_2 = self._get_corpora(target_class)

        # Step 2: Calculate word frequencies in each corpus
        freqs_1 = self._calculate_word_frequencies(corpus_1)
        freqs_2 = self._calculate_word_frequencies(corpus_2)

        # Step 3: Compute log ratios
        log_ratios = self._log_ratio_analysis(freqs_1, freqs_2)

        return log_ratios

    def get_frequencies(self, target_class):
        """
        Returns a dictionary where the key is the word and the value is a tuple:
        (in-class frequency, total frequency).
        """
        # Step 1: Get the two corpora based on the target class
        corpus_1, corpus_2 = self._get_corpora(target_class)

        # Step 2: Calculate word frequencies in each corpus
        freqs_1 = self._calculate_word_frequencies(corpus_1)  # Target class frequencies
        freqs_2 = self._calculate_word_frequencies(corpus_2)  # Non-target class frequencies

        total_freqs = freqs_1 + freqs_2  # Combine both to get total frequencies

        # Step 3: Create a dictionary of word: (in-class frequency, total frequency)
        word_frequencies = {word: (freqs_1.get(word, 0), total_freqs[word]) for word in total_freqs}

        return word_frequencies


def tokens_above_threshold(dataframe, threshold, log_ratio_column="Log_Ratio"):
    """Returns the number of tokens with log ratios greater than or equal to the threshold."""
    return dataframe[abs(dataframe[log_ratio_column]) >= threshold].shape[0]


def generate_thresholds(max_threshold, step):
    """Generates a list of thresholds from 0 to max_threshold with a given step size."""
    return np.arange(0, max_threshold + step, step)


def plot_scree_log_ratios(df, log_ratio_column="Log_Ratio", max_threshold=6, step=0.1):
    """
    Plots a scree plot showing the number of tokens kept based on the log ratio threshold.

    Parameters:
    df (pd.DataFrame): DataFrame containing the log ratios and corresponding words.
    log_ratio_column (str): Name of the column that contains the log ratios.
    max_threshold (float): Maximum threshold value to consider.
    step (float): Step size for the thresholds.
    """
    # Generate thresholds
    thresholds = generate_thresholds(max_threshold, step)

    # Calculate the number of tokens above each threshold
    token_counts = [tokens_above_threshold(df, t, log_ratio_column) for t in thresholds]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(thresholds, token_counts, marker="o")
    plt.title("Scree Plot: Number of Tokens to Keep vs Threshold")
    plt.xlabel("Log Ratio Threshold")
    plt.ylabel("Number of Tokens")
    plt.grid(True)
    plt.show()


def plot_scree_subplots_for_genres(df, text_column, genre_column, max_threshold=6, step=0.05):
    """
    Creates a subplot of scree plots for each genre based on log ratios.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text and genre columns.
    text_column (str): The name of the text column.
    genre_column (str): The column that contains genre labels.
    genres (list): A list of genres to generate scree plots for.
    max_threshold (float): The maximum threshold value.
    step (float): The step size for thresholds.
    """
    genres = df[genre_column].unique()
    num_genres = len(genres)

    # Create subplots with a grid layout
    fig, axs = plt.subplots(num_genres, 1, figsize=(10, 5 * num_genres))

    if num_genres == 1:
        axs = [axs]  # Ensure axs is a list even for a single subplot

    for i, genre in enumerate(genres):
        logratio = LogRatioAnalysis(df, text_column, genre_column)
        log_ratios = logratio.get_log_ratios(genre)
        log_ratios_df = pd.DataFrame(list(log_ratios.items()), columns=["Word", "Log_Ratio"])

        # Generate thresholds and calculate token counts
        thresholds = generate_thresholds(max_threshold, step)
        token_counts = [tokens_above_threshold(log_ratios_df, t) for t in thresholds]

        # Plot the scree plot for the current genre
        axs[i].scatter(thresholds, token_counts, marker="o", label=f"{genre} Tokens")
        axs[i].set_title(f"Scree Plot: {genre}")
        axs[i].set_xlabel("Log Ratio Threshold")
        axs[i].set_ylabel("Number of Tokens")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()
    plt.show()


def plot_log_ratios(df, text_column, genre):
    """
    Plots the scree plot for a specific genre.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text and genre columns.
    text_column (str): The name of the text column.
    genre (str): The genre to analyze.
    """
    logratio = LogRatioAnalysis(df, text_column, "genre")
    log_ratios = logratio.get_log_ratios(genre)
    log_ratios_df = pd.DataFrame(list(log_ratios.items()), columns=["Word", "Log_Ratio"])

    # Plot the scree plot
    plot_scree_log_ratios(log_ratios_df, log_ratio_column="Log_Ratio", max_threshold=6, step=0.1)


def get_log_ratios_df(log_ratio_analysis, target_class):
    """
    Get the top 500 words based on the absolute log ratios and return them as a DataFrame.

    Parameters:
    log_ratio_analysis (LogRatioAnalysis): An instance of LogRatioAnalysis.
    target_class (str): The target class for which to compute log ratios.

    Returns:
    pd.DataFrame: DataFrame containing the top 500 words with columns: 'Word', 'Log_Ratio', 'In_Class_Freq', 'Total_Freq'
    """
    log_ratios = log_ratio_analysis.get_log_ratios(target_class)
    word_frequencies = log_ratio_analysis.get_frequencies(target_class)
    data = []
    for word, log_ratio in log_ratios.items():
        in_class_freq, total_freq = word_frequencies.get(word, (0, 0))
        data.append({"Word": word, "Log_Ratio": log_ratio, "In_Class_Freq": in_class_freq, "Total_Freq": total_freq})
    df = pd.DataFrame(data)
    df["Abs_Log_Ratio"] = df["Log_Ratio"].abs()
    df_sorted = df.sort_values(by="Abs_Log_Ratio", ascending=False)

    return df_sorted[["Word", "Log_Ratio", "In_Class_Freq", "Total_Freq"]]
