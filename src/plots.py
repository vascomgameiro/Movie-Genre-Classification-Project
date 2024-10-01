from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords


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


