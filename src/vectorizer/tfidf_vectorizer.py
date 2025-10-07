import numpy as np
from typing import List
from .base_vectorizer import BaseVectorizer
import os
import pickle


class TFIDFVectorizer(BaseVectorizer):
    def __init__(self, vocabulary_size: int, normalize: bool = True, 
                 smooth_idf: bool = True, **kwargs):
        super().__init__(vocabulary_size, **kwargs)
        self.normalize = normalize
        self.smooth_idf = smooth_idf
        self.idf_ = None
        self.n_documents = 0
    
    def fit(self, vocabulary_indices: List[np.ndarray]) -> 'TFIDFVectorizer':
        self.n_documents = len(vocabulary_indices)
        
        # Calculate document frequencies
        doc_freq = np.zeros(self.vocabulary_size, dtype=np.int32)
        
        for indices in vocabulary_indices:
            if len(indices) > 0:
                unique_indices = np.unique(indices)
                doc_freq[unique_indices] += 1
        
        # Calculate IDF
        if self.smooth_idf:
            # Add 1 to document frequency and total documents
            self.idf_ = np.log((self.n_documents + 1) / (doc_freq + 1)) + 1
        else:
            # Avoid division by zero
            doc_freq = np.maximum(doc_freq, 1)
            self.idf_ = np.log(self.n_documents / doc_freq)
        
        self.is_fitted = True
        return self
    
    def transform(self, vocabulary_indices: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Calculate term frequencies
        tf = np.zeros(self.vocabulary_size, dtype=np.float32)
        
        if len(vocabulary_indices) > 0:
            unique_indices, counts = np.unique(vocabulary_indices, return_counts=True)
            tf[unique_indices] = counts.astype(np.float32)
            
            # Normalize term frequencies
            if self.normalize:
                total_count = np.sum(tf)
                if total_count > 0:
                    tf = tf / total_count
        
        # Apply IDF weighting
        tfidf = tf * self.idf_
        
        return tfidf
    
    def save(self, save_dir: str) -> None:
        """
        Save TFIDFVectorizer to directory.
        
        Args:
            save_dir: Directory to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before saving")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save TF-IDF specific data
        tfidf_data = {
            'idf_': self.idf_,
            'n_documents': self.n_documents,
            'normalize': self.normalize,
            'smooth_idf': self.smooth_idf
        }
        tfidf_path = os.path.join(save_dir, 'tfidf_data.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf_data, f)
        
        # Save base vectorizer parameters
        super().save(save_dir)
    
    def load(self, save_dir: str) -> None:
        """
        Load TFIDFVectorizer from directory.
        
        Args:
            save_dir: Directory containing the saved vectorizer
        """
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Vectorizer directory not found: {save_dir}")
        
        # Load TF-IDF specific data
        tfidf_path = os.path.join(save_dir, 'tfidf_data.pkl')
        with open(tfidf_path, 'rb') as f:
            tfidf_data = pickle.load(f)
            self.idf_ = tfidf_data['idf_']
            self.n_documents = tfidf_data['n_documents']
            self.normalize = tfidf_data['normalize']
            self.smooth_idf = tfidf_data['smooth_idf']
        
        # Load base vectorizer parameters
        super().load(save_dir)
