import math
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd


class LogRatioAnalysis:
    def __init__(self, df, column, target_column, tokenizer=None):
        self.df = df
        self.column = column
        self.target_column = target_column
        self.tokenizer = tokenizer if tokenizer else nltk.word_tokenize
        self.log_ratios_cache = {}  # Cache log ratios for each class
        self.target_classes = df[target_column].unique()
        
    def _calculate_word_frequencies(self, corpus):
        word_counts = Counter()
        for document in corpus:
            if isinstance(document, str):
                tokens = self.tokenizer(document)
                word_counts.update(tokens)
        return word_counts

    def _get_corpora(self, target_class):
        corpus_1 = self.df[self.df[self.target_column] == target_class][self.column].tolist()
        corpus_2 = self.df[self.df[self.target_column] != target_class][self.column].tolist()
        return corpus_1, corpus_2

    def _log_ratio_analysis(self, freqs_1, freqs_2):
        log_ratios = {}
        total_1 = sum(freqs_1.values())
        total_2 = sum(freqs_2.values())

        for word in set(freqs_1.keys()).union(freqs_2.keys()):
            p1 = (freqs_1.get(word, 0) + 1) / total_1
            p2 = (freqs_2.get(word, 0) + 1) / total_2
            log_ratios[word] = math.log2(p1 / p2)

        return log_ratios

    def get_log_ratios(self, target_class):
        """
        Compute and return log ratios for the target class. Cache the result to avoid recomputation.
        """
        if target_class in self.log_ratios_cache:
            return self.log_ratios_cache[target_class]

        corpus_1, corpus_2 = self._get_corpora(target_class)
        freqs_1 = self._calculate_word_frequencies(corpus_1)
        freqs_2 = self._calculate_word_frequencies(corpus_2)
        log_ratios = self._log_ratio_analysis(freqs_1, freqs_2)

        # Cache the result
        self.log_ratios_cache[target_class] = log_ratios
        return log_ratios

    def get_frequencies(self, target_class):
        corpus_1, corpus_2 = self._get_corpora(target_class)
        freqs_1 = self._calculate_word_frequencies(corpus_1)
        freqs_2 = self._calculate_word_frequencies(corpus_2)
        total_freqs = freqs_1 + freqs_2
        return {word: (freqs_1.get(word, 0), total_freqs[word]) for word in total_freqs}

    def feature_selection(self, n, max_threshold=6, step=0.1):
        """
        Perform feature selection by selecting the top `n` tokens for each class.
        """
        thresholds = generate_thresholds(max_threshold, step)
        selected_tokens_set = set()

        for class_label in self.target_classes:
            log_ratios = self.get_log_ratios(class_label)
            log_ratios_df = pd.DataFrame(list(log_ratios.items()), columns=["Token", "Log_Ratio"])
            best_threshold = max_threshold
            for threshold in thresholds:
                token_count = tokens_above_threshold(log_ratios_df, threshold)
                if token_count < n:
                    best_threshold = threshold
                    break

            selected_tokens = log_ratios_df[abs(log_ratios_df["Log_Ratio"]) >= best_threshold]["Token"].tolist()
            selected_tokens_set.update(selected_tokens)

        return selected_tokens_set

def tokens_above_threshold(dataframe, threshold, log_ratio_column="Log_Ratio"):
    """Returns the number of tokens with log ratios greater than or equal to the threshold."""
    return dataframe[abs(dataframe[log_ratio_column]) >= threshold].shape[0]


def generate_thresholds(max_threshold, step):
    """Generates a list of thresholds from 0 to max_threshold with a given step size."""
    return np.arange(0, max_threshold + step, step)

def plot_scree_subplots_for_genres(LogRatio: LogRatioAnalysis, max_threshold=6, step=0.05):
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

    genres = LogRatio.target_classes
    num_genres = len(genres)

    # Create subplots with a grid layout
    fig, axs = plt.subplots(num_genres, 1, figsize=(10, 5 * num_genres))

    if num_genres == 1:
        axs = [axs]  # Ensure axs is a list even for a single subplot

    for i, genre in enumerate(genres):
        log_ratios = LogRatio.get_log_ratios(genre)
        log_ratios_df = pd.DataFrame(list(log_ratios.items()), columns=["Token", "Log_Ratio"])

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
