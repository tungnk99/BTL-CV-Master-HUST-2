import os
import pandas as pd
import numpy as np
import cv2
from typing import List, Tuple


class CifarDataset:
    def __init__(
            self, csv_path: str,
            root_dir: str = None,
            target_size: Tuple[int, int] | None = (32, 32),
            normalize: bool = True
    ):
        self.csv_path = csv_path
        self.root_dir = root_dir if root_dir is not None else ""
        self.target_size = target_size
        self.normalize = normalize
        
        self.data_frame = pd.read_csv(csv_path)
        
        self.class_names = sorted(self.data_frame['label_name'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Loaded dataset with {len(self.data_frame)} samples")
        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {len(self.class_names)}")
    
    def __len__(self) -> int:
        return len(self.data_frame)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:

        if isinstance(idx, list):
            return [self[i] for i in idx]
        
        img_path = self.data_frame.iloc[idx]['file_path']
        label_name = self.data_frame.iloc[idx]['label_name']
        label_idx = self.data_frame.iloc[idx]['label']
        
        if self.root_dir:
            full_img_path = os.path.join(self.root_dir, img_path)
        else:
            full_img_path = img_path
        
        image = self.preprocess(full_img_path)
        
        return image, label_idx, label_name
    
    def get_batch(self, indices: List[int]) -> Tuple[List[np.ndarray], List[int], List[str]]:

        images = []
        labels = []
        names = []
        
        for idx in indices:
            image, label, name = self[idx]
            images.append(image)
            labels.append(label)
            names.append(name)
        
        return images, labels, names
    
    def preprocess(self, image_path: str) -> np.ndarray:

        try:
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Chỉ resize nếu target_size được chỉ định
            if self.target_size is not None:
                image = cv2.resize(image, self.target_size)
            
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.uint8)  # Giữ nguyên uint8 nếu không normalize
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")

            # Tạo fallback image với kích thước phù hợp
            if self.target_size is not None:
                fallback_size = (self.target_size[1], self.target_size[0], 3)
            else:
                fallback_size = (32, 32, 3)  # Default size
                
            if self.normalize:
                fallback = np.zeros(fallback_size, dtype=np.float32)
            else:
                fallback = np.zeros(fallback_size, dtype=np.uint8)
            return fallback
    
    def get_all_data(self) -> Tuple[List[np.ndarray], List[int], List[str]]:
        images = []
        labels = []
        names = []
        
        for idx in range(len(self.data_frame)):
            image, label, name = self[idx]
            images.append(image)
            labels.append(label)
            names.append(name)
        
        return images, labels, names
    
    def get_class_weights(self) -> List[float]:

        class_counts = self.data_frame['label'].value_counts().sort_index()
        total_samples = len(self.data_frame)
        num_classes = len(self.class_names)
        
        weights = []
        for i in range(num_classes):
            if i in class_counts.index:
                weight = total_samples / (num_classes * class_counts[i])
                weights.append(weight)
            else:
                weights.append(0.0)
        
        return weights
    
    def get_class_distribution(self) -> dict:
        return self.data_frame['label_name'].value_counts().to_dict()
    
    def get_sample_by_class(self, class_name: str, n_samples: int = 5) -> List[Tuple[np.ndarray, int, str]]:
        class_data = self.data_frame[self.data_frame['label_name'] == class_name]
        
        if len(class_data) == 0:
            print(f"No samples found for class: {class_name}")
            return []
        
        # Sample n_samples indices
        sample_indices = np.random.choice(len(class_data), 
                                        min(n_samples, len(class_data)), 
                                        replace=False)
        
        samples = []
        for idx in sample_indices:
            global_idx = class_data.index[idx]
            sample = self[global_idx]
            samples.append(sample)
        
        return samples
    
    def split_dataset(self, train_ratio: float = 0.8, random_state: int = 42) -> Tuple['CifarDataset', 'CifarDataset']:

        np.random.seed(random_state)
        
        indices = np.arange(len(self.data_frame))
        np.random.shuffle(indices)
        
        split_point = int(len(indices) * train_ratio)
        
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        
        train_df = self.data_frame.iloc[train_indices].reset_index(drop=True)
        train_csv_path = self.csv_path.replace('.csv', '_train_split.csv')
        train_df.to_csv(train_csv_path, index=False)
        train_dataset = CifarDataset(train_csv_path, self.root_dir, self.target_size, self.normalize)
        
        val_df = self.data_frame.iloc[val_indices].reset_index(drop=True)
        val_csv_path = self.csv_path.replace('.csv', '_val_split.csv')
        val_df.to_csv(val_csv_path, index=False)
        val_dataset = CifarDataset(val_csv_path, self.root_dir, self.target_size, self.normalize)
        
        return train_dataset, val_dataset


class CifarDataLoader:
    @staticmethod
    def load_datasets(train_csv: str, test_csv: str, root_dir: str = None, 
                     target_size: Tuple[int, int] = (32, 32), normalize: bool = True) -> Tuple[CifarDataset, CifarDataset]:

        train_dataset = CifarDataset(train_csv, root_dir, target_size, normalize)
        test_dataset = CifarDataset(test_csv, root_dir, target_size, normalize)
        
        return train_dataset, test_dataset
    
    @staticmethod
    def create_batch_generator(dataset: CifarDataset, batch_size: int = 32, 
                              shuffle: bool = True, random_state: int = 42):

        indices = np.arange(len(dataset))
        
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            images, labels, names = dataset.get_batch(batch_indices)
            yield images, labels, names
