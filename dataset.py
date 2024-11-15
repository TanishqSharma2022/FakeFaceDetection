import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RealvsFakeDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        """
        Args:
            dataset_dir (str): Path to the dataset directory containing 'real' and 'fake' folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        # Get image file paths for both real and fake
        self.real_images = [os.path.join(dataset_dir, 'real', fname) for fname in os.listdir(os.path.join(dataset_dir, 'real')) if fname.endswith(('.jpg', '.jpeg', '.png'))]
        self.fake_images = [os.path.join(dataset_dir, 'fake', fname) for fname in os.listdir(os.path.join(dataset_dir, 'fake')) if fname.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Combine real and fake images into one list
        self.images_filepaths = self.real_images + self.fake_images

    def __len__(self):
        # Total number of images in the dataset
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        # Get image filepath and label
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Set label: 1.0 for 'real', 0.0 for 'fake'
        label = 1.0 if 'real' in image_filepath else 0.0
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label


# Example of defining transformations with Albumentations
def get_transform():
    return A.Compose([
        A.RandomResizedCrop(width=256, height=256, scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.CLAHE(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),  # Normalization for ImageNet-based models
        ToTensorV2()  # Convert to PyTorch tensor
    ])

# Usage example
if __name__ == '__main__':
    dataset_dir = './dataset/'  # Change to your dataset path
    transform = get_transform()
    
    # Initialize dataset
    dataset = RealvsFakeDataset(dataset_dir=dataset_dir, transform=transform)
    
    # Test loading a sample
    image, label = dataset[0]
    print(f"Image shape: {image.shape}, Label: {label}")
