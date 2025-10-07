import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from .base_classifier import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        # Initialize Random Forest
        self.rf = SKRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        self.classes_ = np.unique(y)
        self.rf.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")
        
        return self.rf.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict_proba")
        
        return self.rf.predict_proba(X)
    
    def get_feature_importances(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before getting feature importances")
        
        return self.rf.feature_importances_
