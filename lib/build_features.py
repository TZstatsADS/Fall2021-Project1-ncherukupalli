"""
Filename: build_features.py
Author: Nikhil Cherukupalli
"""

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def remove_stopwords(word_tokens: list) -> list:
    stop_words = set(stopwords.words('english'))
    return [w for w in word_tokens if w.lower() not in stop_words]


def lem_sentence(word_tokens: list) -> list:
    model = WordNetLemmatizer()
    return [model.lemmatize(word) for word in word_tokens]


def get_clean_sentence(word_tokens: list) -> str:
    tokens_no_stop_words = remove_stopwords(word_tokens)
    tokens_lemmatized = lem_sentence(tokens_no_stop_words)
    return " ".join(tokens_lemmatized)
