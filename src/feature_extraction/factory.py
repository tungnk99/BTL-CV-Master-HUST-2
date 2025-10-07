from typing import Dict, Any
from .base_extractor import BaseFeatureExtractor
from .sift_extractor import SIFTExtractor
from .surf_extractor import SURFExtractor
from .orb_extractor import ORBExtractor


def factory_feature_extractor(method: str, **kwargs) -> BaseFeatureExtractor:

    if method == 'sift':
        return SIFTExtractor(**kwargs)
    elif method == 'surf':
        return SURFExtractor(**kwargs)
    elif method == 'orb':
        return ORBExtractor(**kwargs)
    else:
        available_methods = ['sift', 'surf', 'orb']
        raise ValueError(f"Unknown feature extractor method: {method}. Available methods: {available_methods}")
