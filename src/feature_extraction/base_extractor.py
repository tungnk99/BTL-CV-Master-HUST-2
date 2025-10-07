import os
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional


class BaseFeatureExtractor(ABC):
    def __init__(self, debug: bool = False, log_dir: str = "logs_img", **kwargs):
        self.params = kwargs
        self.debug = debug
        self.log_dir = log_dir

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    @abstractmethod
    def extract_features(self, image: np.ndarray, image_name: str = "") -> np.ndarray:
        pass
    
    def extract_features_batch(self, images: List[np.ndarray], image_names: list[str] = None) -> List[np.ndarray]:
        features_list = []
        for image in images:
            features = self.extract_features(image)
            if features is not None and len(features) > 0:
                features_list.append(features)
        return features_list
    
    def get_feature_dim(self) -> int:
        # Default implementation - should be overridden by subclasses
        return 128
