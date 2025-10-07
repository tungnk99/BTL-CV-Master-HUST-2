import numpy as np
from typing import List
from .base_vectorizer import BaseVectorizer


class CountVectorizer(BaseVectorizer):
    def __init__(self, vocabulary_size: int, normalize: bool = True, **kwargs):
        super().__init__(vocabulary_size, **kwargs)
        self.normalize = normalize
    
    def fit(self, vocabulary_indices: List[np.ndarray]) -> 'CountVectorizer':
        self.is_fitted = True
        return self
    
    def transform(self, vocabulary_indices: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Create vector
        vector = np.zeros(self.vocabulary_size, dtype=np.float32)
        
        if len(vocabulary_indices) > 0:
            # Count occurrences of each visual word
            unique_indices, counts = np.unique(vocabulary_indices, return_counts=True)
            vector[unique_indices] = counts.astype(np.float32)
            
            # Normalize if requested
            if self.normalize:
                total_count = np.sum(vector)
                if total_count > 0:
                    vector = vector / total_count
        
        return vector
