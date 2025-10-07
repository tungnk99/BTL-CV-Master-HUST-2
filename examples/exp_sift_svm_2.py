import os

import numpy as np
import argparse

from tqdm import tqdm
from src import BoVWPipeline, Config
from src.dataset.cifar_dataset import CifarDataset
from src.utils.evaluator import Evaluator


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SIFT Optimized Experiment')
    parser.add_argument('--max_train', type=int, default=50000, 
                       help='Maximum number of training samples (default: 50000)')
    parser.add_argument('--max_test', type=int, default=10000,
                       help='Maximum number of test samples (default: 10000)')
    parser.add_argument('--full_dataset', action='store_true',
                       help='Use full dataset (ignore max limits)')
    
    args = parser.parse_args()
    
    # Configuration cho số lượng samples
    if args.full_dataset:
        MAX_TRAIN_SAMPLES = float('inf')
        MAX_TEST_SAMPLES = float('inf')
    else:
        MAX_TRAIN_SAMPLES = args.max_train
        MAX_TEST_SAMPLES = args.max_test
    
    print("=" * 60)
    print("THỰC NGHIỆM SIFT TỐI ƯU CHO CIFAR-10 (32x32)")
    print("=" * 60)

    if args.full_dataset:
        print("\nUsing FULL DATASET (no limits)")
    else:
        print(f"\nMax train samples: {MAX_TRAIN_SAMPLES}")
        print(f"Max test samples: {MAX_TEST_SAMPLES}")
    print("=" * 60)
    
    # Tạo config cho SIFT TỐI ƯU
    config = Config(
        # General settings
        train_ratio=0.8,
        random_state=42,
        target_size=(64, 64),  # RESIZE LÊN 64x64 - RẤT QUAN TRỌNG!
        save_dir="output/sift_optimized",
        
        # Feature extractor: SIFT TỐI ƯU
        feature_extractor='sift',
        feature_extractor_params={
            'nfeatures': 0,
            'nOctaveLayers': 3,
            'contrastThreshold': 0.015,  # Giảm từ 0.04 -> nhiều keypoints hơn
            'edgeThreshold': 5,           # Giảm từ 10 -> chấp nhận edge yếu hơn
            'sigma': 1.2                  # Giảm từ 1.6 cho ảnh nhỏ
        },
        
        # Vocabulary: K-means (Optimized with MiniBatch) - TĂNG SIZE
        vocabulary_method='kmeans',
        vocabulary_size=2000,  # Tăng từ 1000
        vocabulary_params={
            'n_clusters': 2000,
            'random_state': 42,
            'max_iter': 100,
            'n_init': 3,
            'algorithm': 'lloyd',
            'use_minibatch': True,
            'max_samples': 150000  # Tăng từ 100k
        },
        
        # Vectorizer: TF-IDF (tốt hơn Count cho visual words)
        vectorizer_method='tfidf',
        vectorizer_params={
            'normalize': True,
            'smooth_idf': True
        },
        
        # Classifier: SVM TỐI ƯU
        classifier='svm',
        classifier_params={
            'C': 10.0,           # Tăng từ 1.0
            'kernel': 'linear',  # Linear thường tốt hơn cho high-dim features
            'probability': False,
            'random_state': 42
        }
    )
    
    # Tạo dataset
    print("\nLoading CIFAR-10 dataset...")
    train_dataset = CifarDataset(csv_path='data/dataset/train.csv')
    test_dataset = CifarDataset(csv_path='data/dataset/test.csv')
    
    # Load all data
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    print("Loading training data...")
    train_limit = min(len(train_dataset), MAX_TRAIN_SAMPLES)
    for i in tqdm(range(train_limit), desc="Loading train data"):
        image, label, _ = train_dataset[i]
        train_images.append(image)
        train_labels.append(label)
    
    print("Loading test data...")
    test_limit = min(len(test_dataset), MAX_TEST_SAMPLES)
    for i in tqdm(range(test_limit), desc="Loading test data"):
        image, label, _ = test_dataset[i]
        test_images.append(image)
        test_labels.append(label)
    
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Classes: {len(np.unique(train_labels))}")
    
    # Tạo pipeline
    print("\nInitializing SIFT OPTIMIZED pipeline...")
    pipeline = BoVWPipeline(config)
    
    # Training
    print("\nTraining pipeline...")
    print("This may take several minutes...")
    pipeline.fit(train_images, train_labels)
    
    # Prediction
    print("\nMaking predictions...")
    predictions = pipeline.predict(test_images)
    
    # Evaluation
    print("\nEvaluating results...")
    metrics = Evaluator.evaluate(test_labels, predictions)
    
    # Đánh giá chi tiết
    from sklearn.metrics import classification_report, confusion_matrix
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    report = classification_report(test_labels, predictions, target_names=class_names, output_dict=True)
    cm = confusion_matrix(test_labels, predictions)
    
    # Hiển thị kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ SIFT TỐI ƯU")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro): {metrics['recall']:.4f}")
    print(f"F1-score (macro): {metrics['f1_score']:.4f}")
    
    # Lưu kết quả (tương tự exp_sift_svm.py)
    print(f"\nSaving results to {config.save_dir}...")
    pipeline.save_pipeline(config.save_dir)
    
    import json
    from datetime import datetime
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    results = {
        'metrics': {
            'overall': metrics
        }
    }
    
    with open(f"{config.save_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Lưu confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - SIFT Optimized', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{config.save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Hoàn thành!")
    print(f"Kết quả được lưu tại: {config.save_dir}")


if __name__ == "__main__":
    main()

