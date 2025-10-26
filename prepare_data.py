# Prepare brain tumor MRI dataset for training

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def organize_brain_tumor_data():
    # Organize the brain tumor dataset into train/val/test splits
    print("=" * 60)
    print("ORGANIZING BRAIN TUMOR DATASET")
    print("=" * 60)

    # Define paths
    raw_train = Path('data/raw/Training')
    raw_test = Path('data/raw/Testing')

    # Create organized structure
    organized_dir = Path('data/organized')
    train_dir = organized_dir / 'train'
    val_dir = organized_dir / 'val'
    test_dir = organized_dir / 'test'

    # Check if raw data exists
    if not raw_train.exists():
        print(f"Error: {raw_train} not found!")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print(f"\nExtract to: {Path('data/raw').absolute()}")
        return False

    # Get class names
    classes = [d.name for d in raw_train.iterdir() if d.is_dir()]
    print(f"\nFound {len(classes)} classes: {classes}")

    # Count images
    print("\nCounting images...")
    for class_name in classes:
        train_count = len(list((raw_train / class_name).glob('*.jpg'))) + \
                      len(list((raw_train / class_name).glob('*.png')))
        test_count = len(list((raw_test / class_name).glob('*.jpg'))) + \
                     len(list((raw_test / class_name).glob('*.png')))
        print(f"  {class_name:15s}: {train_count:4d} train, {test_count:4d} test")

    # Create directories
    for split_dir in [train_dir, val_dir, test_dir]:
        for class_name in classes:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("CREATING TRAIN/VAL SPLIT (80/20)")
    print("=" * 60)

    # Process training data -> train/val split
    for class_name in classes:
        print(f"\nProcessing {class_name}...")

        # Get all images from training set
        source_dir = raw_train / class_name
        all_images = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))

        # Split into train/val
        train_images, val_images = train_test_split(
            all_images,
            test_size=0.2,
            random_state=42
        )

        print(f"  Copying {len(train_images)} to train...")
        for img_path in tqdm(train_images, desc="  Train"):
            shutil.copy2(img_path, train_dir / class_name / img_path.name)

        print(f"  Copying {len(val_images)} to val...")
        for img_path in tqdm(val_images, desc="  Val"):
            shutil.copy2(img_path, val_dir / class_name / img_path.name)

    print("\n" + "=" * 60)
    print("COPYING TEST SET")
    print("=" * 60)

    # Copy test data
    for class_name in classes:
        print(f"\nProcessing {class_name}...")
        source_dir = raw_test / class_name
        all_images = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))

        print(f"  Copying {len(all_images)} to test...")
        for img_path in tqdm(all_images, desc="  Test"):
            shutil.copy2(img_path, test_dir / class_name / img_path.name)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for class_name in classes:
        train_count = len(list((train_dir / class_name).glob('*')))
        val_count = len(list((val_dir / class_name).glob('*')))
        test_count = len(list((test_dir / class_name).glob('*')))
        total = train_count + val_count + test_count

        print(f"\n{class_name}:")
        print(f"  Train: {train_count:4d} ({train_count / total * 100:5.1f}%)")
        print(f"  Val:   {val_count:4d} ({val_count / total * 100:5.1f}%)")
        print(f"  Test:  {test_count:4d} ({test_count / total * 100:5.1f}%)")
        print(f"  Total: {total:4d}")

    total_train = sum(len(list((train_dir / c).glob('*'))) for c in classes)
    total_val = sum(len(list((val_dir / c).glob('*'))) for c in classes)
    total_test = sum(len(list((test_dir / c).glob('*'))) for c in classes)

    print(f"\nOverall Total:")
    print(f"  Train: {total_train}")
    print(f"  Val:   {total_val}")
    print(f"  Test:  {total_test}")
    print(f"  Total: {total_train + total_val + total_test}")

    print("\n" + "=" * 60)
    print("✓ DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nYour data is ready at:")
    print(f"  Train: {train_dir.absolute()}")
    print(f"  Val:   {val_dir.absolute()}")
    print(f"  Test:  {test_dir.absolute()}")

    return True


if __name__ == '__main__':
    success = organize_brain_tumor_data()

    if success:
        print("\n✓ Ready to start training!")
        print("\nNext step:")
        print("  python train.py --data-dir data/organized --epochs 25")
    else:
        print("\nPlease download the dataset first")