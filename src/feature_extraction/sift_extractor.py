"""
SIFT feature extractor implementation.
"""
import cv2
import numpy as np
from typing import Optional
from .base_extractor import BaseFeatureExtractor


class SIFTExtractor(BaseFeatureExtractor):
    def __init__(
            self, 
            n_features: int = 0, 
            n_octave_layers: int = 3, 
            contrast_threshold: float = 0.04,
            edge_threshold: float = 10,
            sigma: float = 1.6, **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
    
    def extract_features(self, image: np.ndarray, image_name: str = "") -> Optional[np.ndarray]:
        # Check if image is valid
        if image is None or image.size == 0:
            print(f"Warning: Empty or invalid image: {image_name}")
            return np.array([]).reshape(0, 128)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume image is in range [0, 1] and convert to [0, 255]
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to grayscale if necessary
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        elif len(image.shape) == 2:
            gray = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Ensure grayscale image is uint8
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        if self.debug:
            image_name = image_name or "img.png"
            feature_img = cv2.drawKeypoints(gray, keypoints, image)
            cv2.imwrite(f"{self.log_dir}/sift_{image_name}", feature_img)
        
        if descriptors is None:
            return np.array([]).reshape(0, 128)
        
        return descriptors
    
    def get_feature_dim(self) -> int:
        return 128
