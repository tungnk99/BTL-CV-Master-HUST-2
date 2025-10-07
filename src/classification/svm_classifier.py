import numpy as np
from sklearn.svm import SVC
from .base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale',
                 degree: int = 3, probability: bool = False, random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.random_state = random_state
        
        # Initialize SVM
        self.svm = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            probability=probability,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        self.classes_ = np.unique(y)
        self.svm.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")
        
        return self.svm.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict_proba")
        
        if not self.probability:
            raise ValueError("Probability estimates not enabled. Set probability=True")
        
        return self.svm.predict_proba(X)
