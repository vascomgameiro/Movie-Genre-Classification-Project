# 🎬 Movie Genre Classification Project

## 📜 Overview
This project focuses on classifying movie plot descriptions into specific genres using Natural Language Processing (NLP) techniques and machine learning models. The task involves associating a genre to each movie plot from a dataset with genres such as **Drama, Comedy, Horror, Action, Romance, Western, Animation, Crime,** and **Sci-Fi**.

Our approach combines classical NLP techniques with modern embeddings and machine learning models, specifically Histogram Gradient Boosting and Support Vector Classifiers, to create a robust genre classification pipeline.

## 🏆 Project Goals
1. **Genre Classification**: Develop a model to accurately classify movie plots by genre.
2. **Model Evaluation**: Evaluate model performance and aim to exceed baselines:
   - Weak baseline: 37% accuracy
   - Strong baseline: 62% accuracy
3. **Critical Analysis**: Provide detailed analysis of data, model performance, and classification errors.
4. **Future Directions**: Identify potential improvements for future iterations.

## 📊 Data and Preprocessing
### Data Sources
- **Training Set**: Contains labeled plots with metadata (title, director, and genre).
- **Test Set**: Provided without genre labels to evaluate final model predictions.

### Preprocessing Techniques
1. **Director Standardization**: Cleaned and standardized director names.
2. **Text Tokenization and Lemmatization**: Tokenized plot descriptions, expanded contractions, and lemmatized tokens.
3. **Feature Engineering**:
   - Created region-based similarity features for directors.
   - Used TF-IDF vectorization and truncated SVD for dimensionality reduction.

## 🚀 Model Pipeline
### Models Implemented
1. **Histogram Gradient Boosting Classifier** (HGBC) with dense vectors (achieved highest F1 macro-average).
2. **Support Vector Classifier** (SVC) with sparse and dense embeddings.

### Embedding Techniques
- **Sentence Embeddings**: GIST small Embedding model was used for embedding large text, with recursive chunking to handle long descriptions.

### Evaluation
- **Cross-validation** and **Randomized Search** for hyperparameter tuning.
- Metrics: Accuracy, F1 scores per genre, and confusion matrix.

## 🔍 Experimental Results
- **Best Model**: HGBC with sentence embeddings achieved 67.5% accuracy on the test set.
- **Top F1 Scores**: Achieved high F1 scores for genres like Animation and Horror, but struggled with Romance due to plot length variations.

## 📂 Repository Structure
- **`data/`**: Contains training and test datasets.
- **`src/`**: Contains Python scripts for data preprocessing, model training, and evaluation.
- **`notebooks/`**: Contains a Jupyter Notebook describing all the project steps.
- **`README.md`**: Project documentation (you are here!).
- **`NLP_Report.pdf`**: Short paper summarizing the project.
