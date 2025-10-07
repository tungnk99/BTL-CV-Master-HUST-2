from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional
import os
import pickle


class BaseVectorizer(ABC):
    def __init__(self, vocabulary_size: int, **kwargs):
        self.vocabulary_size = vocabulary_size
        self.params = kwargs
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, vocabulary_indices: List[np.ndarray]) -> 'BaseVectorizer':
        pass
    
    @abstractmethod
    def transform(self, vocabulary_indices: np.ndarray) -> np.ndarray:
        pass
    
    def fit_transform(self, vocabulary_indices: List[np.ndarray]) -> List[np.ndarray]:
        self.fit(vocabulary_indices)
        return [self.transform(indices) for indices in vocabulary_indices]
    
    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size
    
    def save(self, save_dir: str) -> None:
        """
        Save vectorizer to directory.
        
        Args:
            save_dir: Directory to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before saving")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save vectorizer parameters
        params = {
            'vocabulary_size': self.vocabulary_size,
            'params': self.params,
            'is_fitted': self.is_fitted
        }
        params_path = os.path.join(save_dir, 'vectorizer_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, save_dir: str) -> None:
        """
        Load vectorizer from directory.
        
        Args:
            save_dir: Directory containing the saved vectorizer
        """
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Vectorizer directory not found: {save_dir}")
        
        # Load vectorizer parameters
        params_path = os.path.join(save_dir, 'vectorizer_params.pkl')
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            self.vocabulary_size = params['vocabulary_size']
            self.params = params['params']
            self.is_fitted = params['is_fitted']
