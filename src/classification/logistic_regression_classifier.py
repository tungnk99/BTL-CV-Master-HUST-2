import numpy as np
from sklearn.linear_model import LogisticRegression
from .base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, C: float = 1.0, penalty: str = 'l2', solver: str = 'lbfgs',
                 max_iter: int = 1000, random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize Logistic Regression
        self.lr = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionClassifier':
        self.classes_ = np.unique(y)
        self.lr.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")
        
        return self.lr.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict_proba")
        
        return self.lr.predict_proba(X)
    
    def get_coefficients(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before getting coefficients")
        
        return self.lr.coef_
