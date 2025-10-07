import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Config:
    # Feature extraction parameters
    feature_extractor: str = 'sift'  # 'sift', 'surf', 'orb'
    feature_extractor_params: Dict[str, Any] = None
    
    # vocabulary parameters
    vocabulary_method: str = 'kmeans'  # 'kmeans'
    vocabulary_size: int = 1000
    vocabulary_params: Dict[str, Any] = None
    
    # Vectorization parameters
    vectorizer_method: str = 'count'  # 'count', 'tfidf'
    vectorizer_params: Dict[str, Any] = None
    
    # Classification parameters
    classifier: str = 'svm'  # 'svm', 'random_forest', 'logistic_regression'
    classifier_params: Dict[str, Any] = None
    
    # Data parameters
    train_ratio: float = 0.8
    random_state: int = 42
    target_size: Optional[tuple] = None  # (width, height) for resizing
    
    # Save/Output parameters
    save_dir: str = "output"
    
    def __post_init__(self):
        """Initialize default parameters after object creation."""
        # Feature extractor default params
        if self.feature_extractor_params is None:
            if self.feature_extractor == 'sift':
                self.feature_extractor_params = {
                    'nfeatures': 0,
                    'nOctaveLayers': 3,
                    'contrastThreshold': 0.04,
                    'edgeThreshold': 10,
                    'sigma': 1.6
                }
            elif self.feature_extractor == 'surf':
                self.feature_extractor_params = {
                    'hessianThreshold': 100,
                    'nOctaves': 4,
                    'nOctaveLayers': 3,
                    'extended': False,
                    'upright': False
                }
            elif self.feature_extractor == 'orb':
                self.feature_extractor_params = {
                    'nfeatures': 500,
                    'scaleFactor': 1.2,
                    'nlevels': 8,
                    'edgeThreshold': 31,
                    'firstLevel': 0,
                    'WTA_K': 2,
                    'scoreType': 0,
                    'patchSize': 31,
                    'fastThreshold': 20
                }
        
        # Vocabulary default params
        if self.vocabulary_params is None:
            self.vocabulary_params = {
                'n_clusters': self.vocabulary_size,
                'random_state': self.random_state,
                'max_iter': 100,
                'n_init': 3,
                'algorithm': 'lloyd',
                'use_minibatch': True,
                'max_samples': 100000
            }
        
        # Vectorizer default params
        if self.vectorizer_params is None:
            if self.vectorizer_method == 'count':
                self.vectorizer_params = {
                    'normalize': True
                }
            elif self.vectorizer_method == 'tfidf':
                self.vectorizer_params = {
                    'normalize': True,
                    'smooth_idf': True
                }
        
        # Classifier default params
        if self.classifier_params is None:
            if self.classifier == 'svm':
                self.classifier_params = {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'degree': 3,
                    'probability': False,
                    'random_state': self.random_state
                }
            elif self.classifier == 'random_forest':
                self.classifier_params = {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'random_state': self.random_state
                }
            elif self.classifier == 'logistic_regression':
                self.classifier_params = {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                    'random_state': self.random_state
                }
        
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'feature_extractor': self.feature_extractor,
            'vocabulary_method': self.vocabulary_method,
            'vectorizer_method': self.vectorizer_method,
            'classifier': self.classifier,
            'vocabulary_size': self.vocabulary_size,
            'train_ratio': self.train_ratio,
            'random_state': self.random_state,
            'target_size': self.target_size,
            'save_dir': self.save_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Config object loaded from YAML
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # Handle new YAML structure with GENERAL and PIPELINE sections
        if 'GENERAL' in yaml_data and 'PIPELINE' in yaml_data:
            general = yaml_data['GENERAL']
            pipeline = yaml_data['PIPELINE']
            
            # Extract configuration
            config_dict = {
                'train_ratio': general.get('train_ratio', 0.8),
                'random_state': general.get('random_state', 42),
                'target_size': general.get('target_size'),
                'save_dir': general.get('save_dir', 'output'),
                
                'feature_extractor': pipeline['feature_extractor']['method'],
                'feature_extractor_params': pipeline['feature_extractor']['params'],
                
                'vocabulary_method': pipeline['vocabulary']['method'],
                'vocabulary_size': pipeline['vocabulary']['size'],
                'vocabulary_params': pipeline['vocabulary']['params'],
                
                'vectorizer_method': pipeline['vectorizer']['method'],
                'vectorizer_params': pipeline['vectorizer']['params'],
                
                'classifier': pipeline['classifier']['method'],
                'classifier_params': pipeline['classifier']['params']
            }
        else:
            # Handle old YAML structure (backward compatibility)
            config_dict = yaml_data
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save the YAML configuration file
        """
        # Create new YAML structure
        yaml_data = {
            'GENERAL': {
                'train_ratio': self.train_ratio,
                'random_state': self.random_state,
                'target_size': self.target_size,
                'save_dir': self.save_dir
            },
            'PIPELINE': {
                'feature_extractor': {
                    'method': self.feature_extractor,
                    'params': self.feature_extractor_params
                },
                'vocabulary': {
                    'method': self.vocabulary_method,
                    'size': self.vocabulary_size,
                    'params': self.vocabulary_params
                },
                'vectorizer': {
                    'method': self.vectorizer_method,
                    'params': self.vectorizer_params
                },
                'classifier': {
                    'method': self.classifier,
                    'params': self.classifier_params
                }
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_config(cls, config_path: str) -> 'Config':
        """
        Load configuration from file (supports both YAML and Python).
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Config object loaded from file
        """
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return cls.from_yaml(config_path)
        elif config_path.endswith('.py'):
            # Load from Python file
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Look for config variables in the module
            if hasattr(config_module, 'DEFAULT_CONFIG'):
                return config_module.DEFAULT_CONFIG
            elif hasattr(config_module, 'config'):
                return config_module.config
            else:
                raise ValueError(f"No valid config found in {config_path}")
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
