import os
import cv2
import numpy as np
from tqdm import tqdm

class CustomDatasetLoader:
    def __init__(self, dataset_dir, image_size=(224, 224)):
        """
        Initialize the dataset loader
        Args:
            dataset_dir (str): Path to dataset directory containing 'real' and 'fake' folders
            image_size (tuple): Target size for images (height, width)
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        
        # Get paths for real and fake images
        self.real_dir = os.path.join(dataset_dir, 'ben')
        self.fake_dir = os.path.join(dataset_dir, 'mal')
        
        # Get all image files
        self.real_files = self._get_image_files(self.real_dir)
        self.fake_files = self._get_image_files(self.fake_dir)
        
        print(f"Found {len(self.real_files)} real images and {len(self.fake_files)} fake images")
    
    def _get_image_files(self, directory):
        """Get all image files from a directory"""
        valid_extensions = ('.jpg', '.jpeg', '.png')
        return [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.lower().endswith(valid_extensions)]
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _apply_augmentation(self, image):
        """Apply basic augmentations to an image"""
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        # Horizontal flip
        flipped = np.fliplr(image)
        augmented_images.append(flipped)
        
        # Brightness adjustment
        brightness = np.clip(image * np.random.uniform(0.8, 1.2), 0, 1)
        augmented_images.append(brightness)
        
        # Contrast adjustment
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        contrast = np.clip((image - mean) * np.random.uniform(0.8, 1.2) + mean, 0, 1)
        augmented_images.append(contrast)


        return augmented_images
    
    def load_dataset(self, with_augmentation=False):
        """
        Load the complete dataset
        Args:
            with_augmentation (bool): Whether to apply data augmentation
        Returns:
            images: numpy array of images
            labels: numpy array of labels (1 for real, 0 for fake)
        """
        images = []
        labels = []
        
        # Load real images
        print("Loading real images...")
        for img_path in tqdm(self.real_files):
            image = self._load_and_preprocess_image(img_path)
            if image is not None:
                if with_augmentation:
                    augmented = self._apply_augmentation(image)
                    images.extend(augmented)
                    labels.extend([1] * len(augmented))
                else:
                    images.append(image)
                    labels.append(1)
        
        # Load fake images
        print("Loading fake images...")
        for img_path in tqdm(self.fake_files):
            image = self._load_and_preprocess_image(img_path)
            if image is not None:
                if with_augmentation:
                    augmented = self._apply_augmentation(image)
                    images.extend(augmented)
                    labels.extend([0] * len(augmented))
                else:
                    images.append(image)
                    labels.append(0)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        
        return images, labels

# Example usage
if __name__ == '__main__':
    # Initialize dataset loader
    dataset_dir = './dataset/'  # Change to your dataset path
    loader = CustomDatasetLoader(dataset_dir, image_size=(224, 224))
    
    # Load simple dataset (without augmentation)
    print("\nLoading simple dataset...")
    images, labels = loader.load_dataset(with_augmentation=False)
    print(f"Simple dataset:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of real images: {np.sum(labels == 1)}")
    print(f"Number of fake images: {np.sum(labels == 0)}")
    
    # Load augmented dataset
    print("\nLoading augmented dataset...")
    aug_images, aug_labels = loader.load_dataset(with_augmentation=True)
    print(f"Augmented dataset:")
    print(f"Images shape: {aug_images.shape}")
    print(f"Labels shape: {aug_labels.shape}")
    print(f"Number of real images: {np.sum(aug_labels == 1)}")
    print(f"Number of fake images: {np.sum(aug_labels == 0)}")
    
    # Example of data ranges and shapes
    print("\nData statistics:")
    print(f"Image value range: [{aug_images.min():.3f}, {aug_images.max():.3f}]")
    print(f"Label values: {np.unique(aug_labels)}")