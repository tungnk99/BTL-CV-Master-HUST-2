import cv2
import numpy as np
from typing import Optional
from .base_extractor import BaseFeatureExtractor


class SURFExtractor(BaseFeatureExtractor):
    def __init__(self, hessianThreshold: float = 100, nOctaves: int = 4,
                 nOctaveLayers: int = 3, extended: bool = False,
                 upright: bool = False, **kwargs):
        super().__init__(**kwargs)
        
        self.hessianThreshold = hessianThreshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers
        self.extended = extended
        self.upright = upright
        
        # Initialize SURF detector
        self.detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessianThreshold,
            nOctaves=nOctaves,
            nOctaveLayers=nOctaveLayers,
            extended=extended,
            upright=upright
        )
    
    def extract_features(self, image: np.ndarray, image_name: str = "") -> Optional[np.ndarray]:
        # Check if image is valid
        if image is None or image.size == 0:
            feature_dim = 128 if self.extended else 64
            print(f"Warning: Empty or invalid image: {image_name}")
            return np.array([]).reshape(0, feature_dim)
        
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
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if self.debug:
            image_name = image_name or "img.png"
            feature_img = cv2.drawKeypoints(gray, keypoints, image)
            cv2.imwrite(f"{self.log_dir}/surf_{image_name}", feature_img)

        if descriptors is None:
            feature_dim = 128 if self.extended else 64
            return np.array([]).reshape(0, feature_dim)
        
        return descriptors
    
    def get_feature_dim(self) -> int:
        return 128 if self.extended else 64
