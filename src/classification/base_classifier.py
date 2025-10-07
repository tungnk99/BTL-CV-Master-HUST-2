from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple
import os
import joblib


class BaseClassifier(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
        self.classes_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test data.

        Args:
            X: Test vectors with shape (n_samples, n_features)

        Returns:
            Predicted class probabilities with shape (n_samples, n_classes)
        """
        # Default implementation - should be overridden by subclasses
        raise NotImplementedError("predict_proba not implemented for this classifier")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before scoring")
        
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_classes(self) -> Optional[np.ndarray]:
        return self.classes_
    
    def save(self, save_dir: str) -> None:
        """
        Save classifier to directory.
        
        Args:
            save_dir: Directory to save the classifier
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before saving")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save classifier model using joblib
        model_path = os.path.join(save_dir, 'classifier.joblib')
        joblib.dump(self, model_path)
    
    def load(self, save_dir: str) -> None:
        """
        Load classifier from directory.
        
        Args:
            save_dir: Directory containing the saved classifier
        """
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Classifier directory not found: {save_dir}")
        
        # Load classifier model
        model_path = os.path.join(save_dir, 'classifier.joblib')
        loaded_classifier = joblib.load(model_path)
        
        # Copy attributes from loaded classifier
        self.__dict__.update(loaded_classifier.__dict__)
