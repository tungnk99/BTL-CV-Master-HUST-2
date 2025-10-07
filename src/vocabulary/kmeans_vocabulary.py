"""
K-means based vocabulary generation with integrated feature extraction.
"""
import numpy as np
from typing import List, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans
from .base_vocabulary import BaseVocabulary
from ..feature_extraction import SIFTExtractor, SURFExtractor, ORBExtractor
import os
import pickle
from tqdm import tqdm


class VisionVocabulary(BaseVocabulary):
    def __init__(self, n_clusters: int = 1000, random_state: int = 42,
                 max_iter: int = 100, n_init: int = 3,
                 algorithm: str = 'auto', feature_extractor: str = 'sift',
                 use_minibatch: bool = True, max_samples: int = 100000,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.algorithm = algorithm
        self.feature_extractor_name = feature_extractor
        self.use_minibatch = use_minibatch
        self.max_samples = max_samples
        
        # Initialize feature extractor
        self.feature_extractor = self._create_feature_extractor(feature_extractor, **kwargs)
        
        # Initialize K-means (use MiniBatchKMeans for speed)
        if use_minibatch:
            self.kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                max_iter=max_iter,
                batch_size=1000,
                n_init=n_init
            )
        else:
            self.kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                max_iter=max_iter,
                n_init=n_init,
                algorithm=algorithm
            )
    
    def _create_feature_extractor(self, feature_extractor: str, **kwargs):
        """Create feature extractor based on name."""
        if feature_extractor == 'sift':
            return SIFTExtractor(**kwargs.get('sift_params', {}))
        elif feature_extractor == 'surf':
            return SURFExtractor(**kwargs.get('surf_params', {}))
        elif feature_extractor == 'orb':
            return ORBExtractor(**kwargs.get('orb_params', {}))
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
    
    def fit(self, images: List[np.ndarray]) -> 'VisionVocabulary':
        """
        Fit the K-means vocabulary to training images.
        Automatically extracts features and builds visual vocabulary.
        
        Args:
            images: List of training images (numpy arrays)
            
        Returns:
            Self for method chaining
        """
        # Step 1: Extract features from all images
        print("Extracting features from images...")
        features = self.feature_extractor.extract_features_batch(images)
        
        # Step 2: Combine all features into a single array
        print("Combining features...")
        if len(features) == 0:
            raise ValueError(f"No features extracted from {len(images)} images! "
                           f"Check if images are valid and feature extractor is working properly.")
        
        all_features = np.vstack(features)
        print(f"Total features extracted: {len(all_features)}")
        
        # Step 3: Subsample features if too many
        if len(all_features) > self.max_samples:
            print(f"Subsampling {len(all_features)} features to {self.max_samples} for faster training...")
            np.random.seed(self.random_state)
            indices = np.random.choice(len(all_features), self.max_samples, replace=False)
            all_features = all_features[indices]
            print(f"Using {len(all_features)} features for vocabulary training")
        
        # Step 4: Fit K-means clustering
        print(f"Training {'MiniBatch' if self.use_minibatch else ''}K-means vocabulary...")
        with tqdm(total=self.max_iter, desc="K-means training") as pbar:
            self.kmeans.fit(all_features)
            pbar.update(self.max_iter)
        
        # Step 5: Store the vocabulary (cluster centers)
        self.vocabulary = self.kmeans.cluster_centers_
        self.is_fitted = True
        
        print("Vocabulary generation completed!")
        
        return self
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Transform image to vocabulary indices using the fitted K-means.
        Automatically extracts features from the image.
        
        Args:
            image: Single image (numpy array)
            
        Returns:
            Array of vocabulary indices
        """
        if not self.is_fitted:
            raise ValueError("Vocabulary must be fitted before transform")
        
        # Extract features from the image
        features = self.feature_extractor.extract_features(image)
        
        if len(features) == 0:
            return np.array([])
        
        # Predict vocabulary assignments
        indices = self.kmeans.predict(features)
        return indices
    
    def get_vocabulary(self) -> Optional[np.ndarray]:
        """
        Get the learned vocabulary (vocabulary centers).
        
        Returns:
            Vocabulary array with shape (n_clusters, feature_dim)
        """
        return self.vocabulary
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Number of visual words in the vocabulary
        """
        return self.n_clusters
    
    def save(self, save_dir: str) -> None:
        """
        Save VisionVocabulary to directory.
        
        Args:
            save_dir: Directory to save the vocabulary
        """
        if not self.is_fitted:
            raise ValueError("Vocabulary must be fitted before saving")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save K-means model
        kmeans_path = os.path.join(save_dir, 'kmeans_model.pkl')
        with open(kmeans_path, 'wb') as f:
            pickle.dump(self.kmeans, f)
        
        # Save vocabulary data
        vocab_path = os.path.join(save_dir, 'vocabulary.pkl')
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.vocabulary, f)
        
        # Save VisionVocabulary specific parameters
        vision_params = {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'n_init': self.n_init,
            'algorithm': self.algorithm,
            'feature_extractor_name': self.feature_extractor_name,
            'use_minibatch': self.use_minibatch,
            'max_samples': self.max_samples,
            'is_fitted': self.is_fitted,
            'params': self.params
        }
        params_path = os.path.join(save_dir, 'vision_vocabulary_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(vision_params, f)
    
    def load(self, save_dir: str) -> None:
        """
        Load VisionVocabulary from directory.
        
        Args:
            save_dir: Directory containing the saved vocabulary
        """
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Vocabulary directory not found: {save_dir}")
        
        # Load K-means model
        kmeans_path = os.path.join(save_dir, 'kmeans_model.pkl')
        with open(kmeans_path, 'rb') as f:
            self.kmeans = pickle.load(f)
        
        # Load vocabulary data
        vocab_path = os.path.join(save_dir, 'vocabulary.pkl')
        with open(vocab_path, 'rb') as f:
            self.vocabulary = pickle.load(f)
        
        # Load VisionVocabulary specific parameters
        params_path = os.path.join(save_dir, 'vision_vocabulary_params.pkl')
        with open(params_path, 'rb') as f:
            vision_params = pickle.load(f)
            self.n_clusters = vision_params['n_clusters']
            self.random_state = vision_params['random_state']
            self.max_iter = vision_params['max_iter']
            self.n_init = vision_params['n_init']
            self.algorithm = vision_params['algorithm']
            self.feature_extractor_name = vision_params['feature_extractor_name']
            self.use_minibatch = vision_params.get('use_minibatch', True)
            self.max_samples = vision_params.get('max_samples', 100000)
            self.is_fitted = vision_params['is_fitted']
            self.params = vision_params['params']
