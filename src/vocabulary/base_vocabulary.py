"""
Base class for vocabulary generation in Bag of Words pipeline.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional
import os
import pickle


class BaseVocabulary(ABC):
    """
    Abstract base class for vocabulary generation.
    
    A vocabulary is a collection of visual words (vocabulary centers) that
    represent the vocabulary of the visual features.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the vocabulary generator.
        
        Args:
            **kwargs: Additional parameters specific to the vocabulary method
        """
        self.params = kwargs
        self.vocabulary = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, images: List[np.ndarray]) -> 'BaseVocabulary':
        """
        Fit the vocabulary to the training images.
        
        Args:
            images: List of training images (numpy arrays)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Transform image to vocabulary indices.
        
        Args:
            image: Single image (numpy array)
            
        Returns:
            Array of vocabulary indices
        """
        pass
    
    def fit_transform(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Fit the vocabulary and transform all images.
        
        Args:
            images: List of training images (numpy arrays)
            
        Returns:
            List of vocabulary index arrays
        """
        self.fit(images)
        return [self.transform(img) for img in images]
    
    def get_vocabulary(self) -> Optional[np.ndarray]:
        """
        Get the learned vocabulary.
        
        Returns:
            Vocabulary array with shape (n_words, feature_dim)
        """
        return self.vocabulary
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary (number of visual words).
        
        Returns:
            Number of visual words in the vocabulary
        """
        if self.vocabulary is None:
            return 0
        return len(self.vocabulary)
    
    def save(self, save_dir: str) -> None:
        """
        Save vocabulary to directory.
        
        Args:
            save_dir: Directory to save the vocabulary
        """
        if not self.is_fitted:
            raise ValueError("Vocabulary must be fitted before saving")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vocabulary data
        vocab_path = os.path.join(save_dir, 'vocabulary.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.vocabulary, f)
        
        # Save vocabulary parameters
        params = {
            'is_fitted': self.is_fitted,
            'params': self.params
        }
        params_path = os.path.join(save_dir, 'vocabulary_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, save_dir: str) -> None:
        """
        Load vocabulary from directory.
        
        Args:
            save_dir: Directory containing the saved vocabulary
        """
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Vocabulary directory not found: {save_dir}")
        
        # Load vocabulary data
        vocab_path = os.path.join(save_dir, 'vocabulary.pkl')
        with open(vocab_path, 'rb') as f:
            self.vocabulary = pickle.load(f)
        
        # Load vocabulary parameters
        params_path = os.path.join(save_dir, 'vocabulary_params.pkl')
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            self.is_fitted = params['is_fitted']
            self.params = params['params']
