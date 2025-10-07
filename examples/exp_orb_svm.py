import os

import numpy as np
from tqdm import tqdm
from src import BoVWPipeline, Config
from src.dataset.cifar_dataset import CifarDataset
from src.utils.evaluator import Evaluator


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ORB + SVM Experiment')
    parser.add_argument('--max_train', type=int, default=50000, 
                       help='Maximum number of training samples (default: 50000)')
    parser.add_argument('--max_test', type=int, default=10000,
                       help='Maximum number of test samples (default: 10000)')
    parser.add_argument('--full_dataset', action='store_true',
                       help='Use full dataset (ignore max limits)')
    
    args = parser.parse_args()
    
    if args.full_dataset:
        MAX_TRAIN_SAMPLES = float('inf')
        MAX_TEST_SAMPLES = float('inf')
    else:
        MAX_TRAIN_SAMPLES = args.max_train
        MAX_TEST_SAMPLES = args.max_test
    
    print("=" * 60)
    print("THỰC NGHIỆM ORB + SVM CHO CIFAR-10")
    print("=" * 60)

    if args.full_dataset:
        print("\nUsing FULL DATASET")
    else:
        print(f"\nMax train samples: {MAX_TRAIN_SAMPLES}")
        print(f"Max test samples: {MAX_TEST_SAMPLES}")
    print("=" * 60)
    
    # Config cho ORB + SVM
    config = Config(
        # General settings
        train_ratio=0.8,
        random_state=42,
        target_size=(64, 64),  # Resize lên 64x64 để có nhiều features hơn
        save_dir="output/orb_svm",
        
        # Feature extractor: ORB TỐI ƯU
        feature_extractor='orb',
        feature_extractor_params={
            'nfeatures': 1000,      # Nhiều features (default 500)
            'scaleFactor': 1.05,     # Nhỏ hơn -> nhiều scale levels
            'nlevels': 4,          # Nhiều pyramid levels
            'edgeThreshold': 5,    # Giảm edge rejection
            'firstLevel': 0,
            'WTA_K': 2,
            'scoreType': 0,         # HARRIS_SCORE
            'patchSize': 15,
            'fastThreshold': 10     # Giảm để có nhiều keypoints
        },
        
        # Vocabulary: MiniBatch K-means
        vocabulary_method='kmeans',
        vocabulary_size=2000,
        vocabulary_params={
            'n_clusters': 2000,
            'random_state': 42,
            'max_iter': 100,
            'n_init': 3,
            'algorithm': 'lloyd',
            'use_minibatch': True,
            'max_samples': 150000
        },
        
        # Vectorizer: TF-IDF
        vectorizer_method='tfidf',
        vectorizer_params={
            'normalize': True,
            'smooth_idf': True
        },
        
        # Classifier: SVM
        classifier='svm',
        classifier_params={
            'C': 10.0,
            'kernel': 'linear',  # Linear tốt cho binary features
            'probability': False,
            'random_state': 42
        }
    )
    
    # Load dataset
    print("\nLoading CIFAR-10 dataset...")
    train_dataset = CifarDataset(csv_path='data/dataset/train.csv')
    test_dataset = CifarDataset(csv_path='data/dataset/test.csv')
    
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
    
    # Training
    print("\nInitializing ORB + SVM pipeline...")
    pipeline = BoVWPipeline(config)
    
    print("\nTraining pipeline...")
    print("ORB nhanh hơn SIFT, ước tính thời gian ngắn hơn...")
    pipeline.fit(train_images, train_labels)
    
    # Prediction
    print("\nMaking predictions...")
    predictions = pipeline.predict(test_images)
    
    # Evaluation
    print("\nEvaluating results...")
    metrics = Evaluator.evaluate(test_labels, predictions)
    
    from sklearn.metrics import classification_report, confusion_matrix
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    report = classification_report(test_labels, predictions, target_names=class_names, output_dict=True)
    cm = confusion_matrix(test_labels, predictions)
    
    # Results
    print("\n" + "=" * 60)
    print("KẾT QUẢ ORB + SVM")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro): {metrics['recall']:.4f}")
    print(f"F1-score (macro): {metrics['f1_score']:.4f}")
    

    # Save
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
    
    # Confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',  # Green cho ORB
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ORB + SVM', fontsize=16, fontweight='bold')
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

