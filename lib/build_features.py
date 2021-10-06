"""
Filename: build_features.py
Author: Nikhil Cherukupalli
"""

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_stopwords(word_tokens: list) -> list:
    """Remove stop words from tokenized string."""
    stop_words = set(stopwords.words('english'))
    return [w for w in word_tokens if w.lower() not in stop_words]


def lem_sentence(word_tokens: list) -> list:
    """Lemmatizes tokenized string."""
    model = WordNetLemmatizer()
    return [model.lemmatize(word) for word in word_tokens]


def get_clean_sentence(word_tokens: list) -> str:
    """Wrapper that removes stop words and lemmatizes tokenized string."""
    tokens_no_stop_words = remove_stopwords(word_tokens)
    tokens_lemmatized = lem_sentence(tokens_no_stop_words)
    return " ".join(tokens_lemmatized)


def get_vectorized_sentences(lst_sentences):
    """ Maps list of strings to a vector space using a count vectorizer.
    :param (list) lst_sentences: Each element is either a string word/sentence.
    :return (np.array):
        Sparse matrix with number of columns equivalent to number of unique
        words across lst_sentences. For example, A_{ij} represents the number
        of occurrences of word j in the ith element of lst_sentences.
    """
    vectorizer = TfidfVectorizer().fit(lst_sentences)
    return vectorizer.transform(lst_sentences)
