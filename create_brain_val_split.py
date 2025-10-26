"""
Create validation split for brain tumor dataset
"""
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path


def create_val_split(train_dir, val_dir, val_ratio=0.2):
    """Split training data into train/val"""

    print("=" * 60)
    print("CREATING VALIDATION SPLIT FOR BRAIN TUMORS")
    print("=" * 60)

    # Check if train_dir exists
    if not os.path.exists(train_dir):
        print(f"ERROR: Training directory not found: {train_dir}")
        return

    print(f"\nSource: {train_dir}")
    print(f"Target: {val_dir}")
    print(f"Split ratio: {val_ratio * 100:.0f}% validation\n")

    # Create validation directory
    os.makedirs(val_dir, exist_ok=True)

    # Process each class
    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_train_path):
            print(f"Skipping {class_name} (not a directory)")
            continue

        class_val_path = os.path.join(val_dir, class_name)
        os.makedirs(class_val_path, exist_ok=True)

        # Get all images (jpg and png)
        images = [f for f in os.listdir(class_train_path)
                  if f.endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) == 0:
            print(f"{class_name}: No images found, skipping")
            continue

        # Split into train/val
        train_imgs, val_imgs = train_test_split(
            images,
            test_size=val_ratio,
            random_state=42
        )

        # Move validation images
        moved_count = 0
        for img in val_imgs:
            src = os.path.join(class_train_path, img)
            dst = os.path.join(class_val_path, img)

            try:
                shutil.move(src, dst)
                moved_count += 1
            except Exception as e:
                print(f"Error moving {img}: {e}")

        print(f"✓ {class_name:15s}: {len(train_imgs):4d} train, {moved_count:4d} val")

    print("\n" + "=" * 60)
    print("VALIDATION SPLIT COMPLETE!")
    print("=" * 60)

    # Summary
    print("\nFinal structure:")
    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)
        class_val_path = os.path.join(val_dir, class_name)

        if os.path.isdir(class_train_path):
            train_count = len([f for f in os.listdir(class_train_path)
                               if f.endswith(('.jpg', '.jpeg', '.png'))])
            val_count = len([f for f in os.listdir(class_val_path)
                             if f.endswith(('.jpg', '.jpeg', '.png'))])

            print(f"  {class_name:15s}: {train_count:4d} train, {val_count:4d} val")


if __name__ == '__main__':
    # YOUR BRAIN TUMOR DATA PATHS
    train_dir = 'data/raw/Training'  # ← Note capital T
    val_dir = 'data/raw/Validation'

    # Check if directory exists
    if not os.path.exists(train_dir):
        print(f"ERROR: Directory not found: {train_dir}")
        print("\nAvailable directories in data/raw:")
        if os.path.exists('data/raw'):
            for item in os.listdir('data/raw'):
                print(f"  - {item}")
        exit(1)

    # Run the split
    create_val_split(train_dir, val_dir, val_ratio=0.2)

    print("\n✓ Done! You can now use these directories:")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")
    print(f"  Test:  data/raw/Testing")