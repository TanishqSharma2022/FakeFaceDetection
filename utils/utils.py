
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np



def get_image_shape_stats(images, labels):
    """Get minimum and maximum image shapes for real and fake images"""
    
    # Separate real and fake images based on labels
    real_images = [img for img, label in zip(images, labels) if label == 1]
    fake_images = [img for img, label in zip(images, labels) if label == 0]
    
    # Get the shapes of the real and fake images
    real_shapes = [img.shape for img in real_images]
    fake_shapes = [img.shape for img in fake_images]
    
    # Get the minimum and maximum shapes for real and fake images
    min_real_shape = np.min(real_shapes, axis=0)
    max_real_shape = np.max(real_shapes, axis=0)
    
    min_fake_shape = np.min(fake_shapes, axis=0)
    max_fake_shape = np.max(fake_shapes, axis=0)
    
    return min_real_shape, max_real_shape, min_fake_shape, max_fake_shape


def plot_confusion_matrix(all_labels, all_preds):
    # Compute confusion matrix using sklearn's confusion_matrix function
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix using seaborn's heatmap for better visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()