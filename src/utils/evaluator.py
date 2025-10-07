import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    @staticmethod
    def evaluate(y_true: List[int], y_pred: List[int], 
                               class_names: List[str] = None) -> Dict[str, float]:
        """
        Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                             class_names: List[str] = None, 
                             save_path: str = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_classification_report(y_true: List[int], y_pred: List[int], 
                                  class_names: List[str] = None) -> None:
        """
        Plot classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        from sklearn.metrics import classification_report
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(set(y_true)))]
        
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)
    
    @staticmethod
    def plot_learning_curve(train_scores: List[float], val_scores: List[float], 
                           save_path: str = None) -> None:
        """
        Plot learning curve.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_scores, label='Training Score', marker='o')
        plt.plot(val_scores, label='Validation Score', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
