
import sys
import os
import numpy as np
from PIL import Image

from src.dataset import CifarDataset

def test_cifar_dataset():
    
    try:
        # Load dataset
        train_csv = 'data/dataset/train.csv'
        
        dataset = CifarDataset(
            csv_path=train_csv,
            target_size=None,  # Không resize, giữ nguyên size
            normalize=False    # Không normalize, giữ nguyên giá trị pixel
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Classes: {dataset.class_names}")
        
        # Test 1 ảnh đầu tiên
        print("\nTesting first image...")
        image, label, class_name = dataset[0]
        
        print(f"  Shape: {image.shape}")
        print(f"  Label: {label}")
        print(f"  Class: {class_name}")
        print(f"  Range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Save processed image
        print("\nSaving processed image...")
        output_dir = "processed_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Image đã là uint8 nếu không normalize
        if image.dtype == np.float32:
            # Nếu là float32 (normalized), convert về uint8
            image_to_save = (image * 255).astype(np.uint8)
        else:
            # Nếu đã là uint8, dùng trực tiếp
            image_to_save = image
        
        # Save with PIL
        pil_image = Image.fromarray(image_to_save)
        output_path = os.path.join(output_dir, f"original_image_{class_name}.png")
        pil_image.save(output_path)
        
        print(f"Saved original image: {output_path}")
        print(f"Original size: {image.shape}")
        
        # Test batch
        print("\nTesting batch...")
        batch_images, batch_labels, batch_names = dataset.get_batch([0, 1])
        print(f"✓ Batch loaded: {len(batch_images)} images")
        
        # Save batch images
        print("Saving batch images...")
        for i, (img, lbl, name) in enumerate(zip(batch_images, batch_labels, batch_names)):
            if img.dtype == np.float32:
                img_to_save = (img * 255).astype(np.uint8)
            else:
                img_to_save = img
                
            pil_img = Image.fromarray(img_to_save)
            batch_path = os.path.join(output_dir, f"original_batch_{i+1}_{name}.png")
            pil_img.save(batch_path)
            print(f"✓ Saved original batch image {i+1}: {batch_path} (size: {img.shape})")
        

    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cifar_dataset()
