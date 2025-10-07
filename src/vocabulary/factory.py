"""
Factory functions for creating vocabulary instances.
"""
from typing import Dict, Any
from .base_vocabulary import BaseVocabulary
from .kmeans_vocabulary import VisionVocabulary


def factory_vocabulary(method: str, **kwargs) -> BaseVocabulary:
    """
    Create a vocabulary instance.
    
    Args:
        method: Vocabulary method ('kmeans', 'vision')
        **kwargs: Additional parameters for the vocabulary
        
    Returns:
        Vocabulary instance
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'kmeans' or method == 'vision':
        return VisionVocabulary(**kwargs)
    else:
        available_methods = ['kmeans', 'vision']
        raise ValueError(f"Unknown vocabulary method: {method}. Available methods: {available_methods}")
