from typing import Dict, Any
from .base_classifier import BaseClassifier
from .svm_classifier import SVMClassifier
from .random_forest_classifier import RandomForestClassifier
from .logistic_regression_classifier import LogisticRegressionClassifier


def factory_classifier(method: str, **kwargs) -> BaseClassifier:
    if method == 'svm':
        return SVMClassifier(**kwargs)
    elif method == 'random_forest' or method == 'rf':
        return RandomForestClassifier(**kwargs)
    elif method == 'logistic_regression' or method == 'lr':
        return LogisticRegressionClassifier(**kwargs)
    else:
        available_methods = ['svm', 'random_forest', 'rf', 'logistic_regression', 'lr']
        raise ValueError(f"Unknown classifier method: {method}. Available methods: {available_methods}")
