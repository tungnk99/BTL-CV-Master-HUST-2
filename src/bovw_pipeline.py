import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
import os
import pickle
import joblib
import json
from datetime import datetime

from .feature_extraction import factory_feature_extractor
from .vocabulary import factory_vocabulary
from .vectorizer import factory_vectorizer
from .classification import factory_classifier
from .utils import DataLoader, Evaluator, Config, logger


class BoVWPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.feature_extractor = self._create_feature_extractor()
        self.vocabulary = self._create_vocabulary()
        self.vectorizer = self._create_vectorizer()
        self.classifier = self._create_classifier()
        
        # Pipeline state
        self.is_fitted = False
        self.training_time = 0.0
        self.vocabulary_generation_time = 0.0
        self.vectorization_time = 0.0
        self.classification_training_time = 0.0
    
    def _create_feature_extractor(self):
        """Create feature extractor based on config."""
        return factory_feature_extractor(self.config.feature_extractor, **self.config.feature_extractor_params)
    
    def _create_vocabulary(self):
        """Create vocabulary generator based on config."""
        # Pass feature extractor and its parameters to VisionVocabulary
        vocab_params = self.config.vocabulary_params.copy()
        vocab_params['feature_extractor'] = self.config.feature_extractor
        return factory_vocabulary(self.config.vocabulary_method, **vocab_params)
    
    def _create_vectorizer(self):
        """Create vectorizer generator based on config."""
        params = self.config.vectorizer_params.copy()
        params['vocabulary_size'] = self.config.vocabulary_size
        return factory_vectorizer(self.config.vectorizer_method, **params)
    
    def _create_classifier(self):
        """Create classifier based on config."""
        return factory_classifier(self.config.classifier, **self.config.classifier_params)
    
    def fit(self, images: List[np.ndarray], labels: List[int]) -> 'BoVWPipeline':
        """
        Fit the pipeline to training data.
        
        Args:
            images: List of training images
            labels: List of training labels
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        self.logger.info("Starting pipeline training...")
        
        # Step 1: Vocabulary generation (includes feature extraction)
        self.logger.info(">>>>>> Step 1: Generating vocabulary...")
        vocabulary_start = time.time()
        self.vocabulary.fit(images)
        self.vocabulary_generation_time = time.time() - vocabulary_start
        self.logger.info(f"Vocabulary generation completed in {self.vocabulary_generation_time:.2f}s")
        
        # Step 2: Vectorization
        self.logger.info(">>>>>> Step 2: Creating vectors...")
        vectorization_start = time.time()
        print("Transforming images to vocabulary indices...")
        vocabulary_indices = [self.vocabulary.transform(img) for img in images]
        print("Creating vectors from vocabulary indices...")
        vectors = self.vectorizer.fit_transform(vocabulary_indices)
        self.vectorization_time = time.time() - vectorization_start
        self.logger.info(f"Vectorization completed in {self.vectorization_time:.2f}s")
        
        # Step 3: Classification training
        self.logger.info(">>>>>> Step 3: Training classifier...")
        classification_start = time.time()
        print("Training classifier...")
        X_train = np.array(vectors)
        y_train = np.array(labels)
        self.classifier.fit(X_train, y_train)
        self.classification_training_time = time.time() - classification_start
        self.logger.info(f"Classification training completed in {self.classification_training_time:.2f}s")
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        self.logger.info(f"Pipeline training completed in {self.training_time:.2f}s")
        
        return self
    
    def predict(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Predict labels for test images.
        
        Args:
            images: List of test images
            
        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        self.logger.info("Starting prediction...")
        
        # Transform to vocabulary indices (includes feature extraction)
        vocabulary_indices = [self.vocabulary.transform(img) for img in images]
        
        # Create vectors
        vectors = [self.vectorizer.transform(indices) for indices in vocabulary_indices]
        
        # Predict
        X_test = np.array(vectors)
        predictions = self.classifier.predict(X_test)
        
        self.logger.info("Prediction completed")
        return predictions
    
    def predict_proba(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for test images.
        
        Args:
            images: List of test images
            
        Returns:
            Array of predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Transform to vocabulary indices (includes feature extraction)
        vocabulary_indices = [self.vocabulary.transform(img) for img in images]
        
        # Create vectors
        vectors = [self.vectorizer.transform(indices) for indices in vocabulary_indices]
        
        # Predict probabilities
        X_test = np.array(vectors)
        probabilities = self.classifier.predict_proba(X_test)
        
        return probabilities
    
    def evaluate(self, images: List[np.ndarray], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the pipeline on test data.
        
        Args:
            images: List of test images
            labels: List of true labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(images)
        return Evaluator.evaluate(labels, predictions)
    

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary of pipeline information
        """
        return {
            'feature_extractor': self.config.feature_extractor,
            'vocabulary_method': self.config.vocabulary_method,
            'vectorizer_method': self.config.vectorizer_method,
            'classifier': self.config.classifier,
            'vocabulary_size': self.config.vocabulary_size,
            'is_fitted': self.is_fitted
        }
    
    def save_pipeline(self, save_dir: str) -> None:
        """
        Save the entire pipeline to a directory.
        
        Args:
            save_dir: Directory to save the pipeline components
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save config as JSON
        config_path = os.path.join(save_dir, 'config.json')
        config_dict = self.config.to_dict()
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # Save each component using their own save methods
        vocab_dir = os.path.join(save_dir, 'vocabulary')
        self.vocabulary.save(vocab_dir)
        
        vectorizer_dir = os.path.join(save_dir, 'vectorizer')
        self.vectorizer.save(vectorizer_dir)
        
        classifier_dir = os.path.join(save_dir, 'classifier')
        self.classifier.save(classifier_dir)
        
        # Save pipeline metadata as JSON
        metadata = {
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'vocabulary_generation_time': self.vocabulary_generation_time,
            'vectorization_time': self.vectorization_time,
            'classification_training_time': self.classification_training_time,
            'save_timestamp': datetime.now().isoformat(),
            'pipeline_info': self.get_pipeline_info()
        }
        
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Pipeline saved to {save_dir}")
    
    @classmethod
    def load_pipeline(cls, save_dir: str) -> 'BoVWPipeline':
        """
        Load a saved pipeline from directory.
        
        Args:
            save_dir: Directory containing the saved pipeline
            
        Returns:
            Loaded BoVWPipeline instance
        """
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Pipeline directory not found: {save_dir}")
        
        # Load config from JSON
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
        
        # Create pipeline instance
        pipeline = cls(config)
        
        # Load each component using their own load methods
        vocab_dir = os.path.join(save_dir, 'vocabulary')
        pipeline.vocabulary.load(vocab_dir)
        
        vectorizer_dir = os.path.join(save_dir, 'vectorizer')
        pipeline.vectorizer.load(vectorizer_dir)
        
        classifier_dir = os.path.join(save_dir, 'classifier')
        pipeline.classifier.load(classifier_dir)
        
        # Load metadata from JSON
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Restore pipeline state
        pipeline.is_fitted = metadata['is_fitted']
        pipeline.training_time = metadata['training_time']
        pipeline.vocabulary_generation_time = metadata['vocabulary_generation_time']
        pipeline.vectorization_time = metadata['vectorization_time']
        pipeline.classification_training_time = metadata['classification_training_time']
        
        pipeline.logger.info(f"Pipeline loaded from {save_dir}")
        pipeline.logger.info(f"Pipeline was saved at: {metadata['save_timestamp']}")
        
        return pipeline


