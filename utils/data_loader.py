import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MedicalImageDataset(Dataset):
    def __init__(self,root_dir,transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
        self.class_to_index = {cls: index for index, cls in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_index = self.class_to_index[class_name]

            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                self.images.append(img_path)
                self.labels.append(class_index)

        print(f"Loaded {len(self.images)} images from {len(self.classes)} classes")
        for cls, index in self.class_to_index.items():
            count = self.labels.count(index)
            print(f" {cls}: {count} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(image_size=224, augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def create_data_loaders(train_dir, val_dir, test_dir, batch_size=32, image_size=224, num_workers=4):
    train_transform, val_transform = get_transforms(image_size, augment=True)
    train_dataset = MedicalImageDataset(train_dir, transform=train_transform)
    val_dataset = MedicalImageDataset(val_dir, transform=val_transform)
    test_dataset = MedicalImageDataset(test_dir, transform=val_transform)

    class_counts = [train_dataset.labels.count(i)
                    for i in range(len(train_dataset.classes))]
    class_weights = [sum(class_counts) / c for c in class_counts]

    print(f"\nClass weights for loss function: {class_weights}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights, train_dataset.classes