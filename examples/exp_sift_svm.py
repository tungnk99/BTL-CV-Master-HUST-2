import os
import numpy as np
import argparse
from tqdm import tqdm
from src import BoVWPipeline, Config
from src.dataset.cifar_dataset import CifarDataset
from src.utils.evaluator import Evaluator


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SIFT + SVM Experiment')
    parser.add_argument('--max_train', type=int, default=50000, 
                       help='Maximum number of training samples (default: 50000)')
    parser.add_argument('--max_test', type=int, default=10000,
                       help='Maximum number of test samples (default: 10000)')
    parser.add_argument('--full_dataset', action='store_true',
                       help='Use full dataset (ignore max limits)')
    
    args = parser.parse_args()
    
    # Configuration cho số lượng samples
    if args.full_dataset:
        MAX_TRAIN_SAMPLES = float('inf')  # Không giới hạn
        MAX_TEST_SAMPLES = float('inf')
    else:
        MAX_TRAIN_SAMPLES = args.max_train
        MAX_TEST_SAMPLES = args.max_test
    
    print("=" * 60)
    print("THỰC NGHIỆM SIFT + SVM (OPTIMIZED)")
    print("=" * 60)
    if args.full_dataset:
        print("Using FULL DATASET (no limits)")
    else:
        print(f"Max train samples: {MAX_TRAIN_SAMPLES}")
        print(f"Max test samples: {MAX_TEST_SAMPLES}")
    print("=" * 60)
    
    # Tạo config cho SIFT + SVM
    config = Config(
        # General settings
        train_ratio=0.8,
        random_state=42,
        target_size=None,
        save_dir="output/sift_svm",
        
        # Feature extractor: SIFT
        feature_extractor='sift',
        feature_extractor_params={
            'nfeatures': 0,
            'nOctaveLayers': 3,
            'contrastThreshold': 0.04,
            'edgeThreshold': 10,
            'sigma': 1.6
        },
        
        # Vocabulary: K-means
        vocabulary_method='kmeans',
        vocabulary_size=1000,
        vocabulary_params={
            'n_clusters': 1000,
            'random_state': 42,
            'max_iter': 100,  # Reduced from 300
            'n_init': 3,      # Reduced from 10
            'algorithm': 'lloyd',
            'use_minibatch': True,  # Use MiniBatchKMeans for speed
            'max_samples': 100000   # Limit features for faster training
        },
        
        # Vectorizer: Count
        vectorizer_method='count',
        vectorizer_params={
            'normalize': True
        },
        
        # Classifier: SVM
        classifier='svm',
        classifier_params={
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'degree': 3,
            'probability': False,
            'random_state': 42
        }
    )
    
    # Tạo dataset
    print("Loading CIFAR-10 dataset...")
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
    print("\nInitializing SIFT + SVM pipeline...")
    pipeline = BoVWPipeline(config)
    
    # Training
    print("\nTraining pipeline...")
    print("This may take several minutes...")
    pipeline.fit(train_images, train_labels)
    
    # Prediction
    print("\nMaking predictions...")
    print("Predicting on test set...")
    predictions = pipeline.predict(test_images)
    
    # Evaluation
    print("\nEvaluating results...")
    metrics = Evaluator.evaluate(test_labels, predictions)
    
    # Đánh giá chi tiết từng class
    from sklearn.metrics import classification_report, confusion_matrix
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Classification report chi tiết
    report = classification_report(test_labels, predictions, target_names=class_names, output_dict=True)
    cm = confusion_matrix(test_labels, predictions)
    
    # Hiển thị kết quả tổng quan
    print("\n" + "=" * 60)
    print("KẾT QUẢ THỰC NGHIỆM SIFT + SVM")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro): {metrics['recall']:.4f}")
    print(f"F1-score (macro): {metrics['f1_score']:.4f}")
    
    # Hiển thị kết quả từng class
    print(f"\n" + "=" * 60)
    print("KẾT QUẢ CHI TIẾT TỪNG CLASS")
    print("=" * 60)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    print("-" * 45)
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            print(f"{class_name:<12} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # Hiển thị confusion matrix
    print(f"\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print("Predicted ->")
    print("Actual ↓", end="")
    for i in range(len(class_names)):
        print(f"{class_names[i][:4]:>6}", end="")
    print()
    
    for i, true_class in enumerate(class_names):
        print(f"{true_class[:4]:<8}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>6}", end="")
        print()
    
    # Lưu model và kết quả
    print(f"\nSaving model and results to {config.save_dir}...")
    pipeline.save_pipeline(config.save_dir)
    
    # Lưu kết quả chi tiết
    import json
    import numpy as np
    from datetime import datetime
    
    # Tạo thư mục output
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Lưu kết quả JSON
    results = {
        'metrics': {
            'overall': metrics,
            'per_class': {class_names[i]: {
                'precision': report[str(i)]['precision'],
                'recall': report[str(i)]['recall'],
                'f1_score': report[str(i)]['f1-score']
            } for i in range(len(class_names)) if str(i) in report}
        },
    }
    
    # Lưu file JSON
    with open(f"{config.save_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Lưu classification report dạng text
    with open(f"{config.save_dir}/classification_report.txt", 'w') as f:
        f.write("SIFT + SVM EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['recall']:.4f}\n")
        f.write(f"Macro F1-score: {metrics['f1_score']:.4f}\n\n")
        
        f.write("PER-CLASS RESULTS:\n")
        f.write("-" * 30 + "\n")
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                f.write(f"{class_name}: P={report[str(i)]['precision']:.4f}, "
                       f"R={report[str(i)]['recall']:.4f}, "
                       f"F1={report[str(i)]['f1-score']:.4f}\n")
    
    # Lưu confusion matrix dạng CSV
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{config.save_dir}/confusion_matrix.csv")
    
    # Lưu confusion matrix dạng ảnh
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - SIFT + SVM', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{config.save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{config.save_dir}/confusion_matrix.pdf", bbox_inches='tight')
    plt.close()
    
    # Lưu predictions
    predictions_df = pd.DataFrame({
        'true_label': test_labels,
        'predicted_label': predictions,
        'correct': [test_labels[i] == predictions[i] for i in range(len(test_labels))]
    })
    predictions_df.to_csv(f"{config.save_dir}/predictions.csv", index=False)
    
    print("Thực nghiệm hoàn thành!")
    print(f"Kết quả được lưu tại: {config.save_dir}")


if __name__ == "__main__":
    main()
