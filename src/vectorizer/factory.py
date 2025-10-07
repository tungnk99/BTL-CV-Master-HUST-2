from typing import Dict, Any
from .base_vectorizer import BaseVectorizer
from .count_vectorizer import CountVectorizer
from .tfidf_vectorizer import TFIDFVectorizer


def factory_vectorizer(method: str, vocabulary_size: int, **kwargs) -> BaseVectorizer:
    if method == 'count':
        return CountVectorizer(vocabulary_size=vocabulary_size, **kwargs)
    elif method == 'tfidf':
        return TFIDFVectorizer(vocabulary_size=vocabulary_size, **kwargs)
    else:
        available_methods = ['count', 'tfidf']
        raise ValueError(f"Unknown vectorizer method: {method}. Available methods: {available_methods}")
