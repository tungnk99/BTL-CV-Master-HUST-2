
import os
import pickle
import numpy as np
from typing import List, Tuple, Optional
import cv2


class DataLoader:
    @staticmethod
    def load_cifar10(data_path: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load CIFAR-10 dataset.
        
        Args:
            data_path: Path to the CIFAR-10 data file
            
        Returns:
            Tuple of (images, labels, class_names)
        """
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Load CIFAR-10 data
        with open(data_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        
        # Extract images and labels
        images = data[b'data']
        labels = data[b'labels']
        
        # Reshape images to (32, 32, 3)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # Convert to list of numpy arrays
        image_list = [images[i] for i in range(len(images))]
        label_list = labels.tolist()
        
        return image_list, label_list, class_names
    
    @staticmethod
    def load_images_from_directory(directory: str, extensions: List[str] = None) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load images from a directory.
        
        Args:
            directory: Path to the directory containing images
            extensions: List of file extensions to load (default: ['.jpg', '.jpeg', '.png', '.bmp'])
            
        Returns:
            Tuple of (images, filenames)
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        images = []
        filenames = []
        
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in extensions):
                filepath = os.path.join(directory, filename)
                image = cv2.imread(filepath)
                if image is not None:
                    images.append(image)
                    filenames.append(filename)
        
        return images, filenames
    
    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Preprocess an image for feature extraction.
        
        Args:
            image: Input image
            target_size: Target size (width, height) for resizing
            
        Returns:
            Preprocessed image
        """
        # Resize if target size is specified
        if target_size is not None:
            image = cv2.resize(image, target_size)
        
        return image
    
    @staticmethod
    def split_data(images: List[np.ndarray], labels: List[int], 
                   train_ratio: float = 0.8, random_state: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
        """
        Split data into training and testing sets.
        
        Args:
            images: List of images
            labels: List of labels
            train_ratio: Ratio of data to use for training
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_images, test_images, train_labels, test_labels)
        """
        np.random.seed(random_state)
        
        # Create indices for shuffling
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        
        # Calculate split point
        split_point = int(len(images) * train_ratio)
        
        # Split indices
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        
        # Split data
        train_images = [images[i] for i in train_indices]
        test_images = [images[i] for i in test_indices]
        train_labels = [labels[i] for i in train_indices]
        test_labels = [labels[i] for i in test_indices]
        
        return train_images, test_images, train_labels, test_labels
