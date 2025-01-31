import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pickle

def compute_mean_std(input_dir):
    """Compute the mean and standard deviation of the input images.

    Args:
        input_dir (str): The directory containing the input images.

    Returns:
        tuple: The mean and standard deviation of the input images.
    """
    # Initialize variables
    pixel_sum = np.zeros(3)
    pixel_squared_sum = np.zeros(3)
    num_pixels = 0

    # Collect all image paths for the progress bar
    image_paths = []
    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for image_name in os.listdir(subfolder_path):
                image_paths.append(os.path.join(subfolder_path, image_name))

    # Iterate through each image with a progress bar
    for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        try:
            image = Image.open(image_path).convert("RGB")
            # Convert to numpy array and normalize
            image_array = np.array(image) / 255.0
            # Compute sum of pixels across height and width
            pixel_sum += np.sum(image_array, axis=(0, 1))
            pixel_squared_sum += np.sum(np.square(image_array), axis=(0, 1))
            num_pixels += image_array.shape[0] * image_array.shape[1]
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    # Compute mean and standard deviation
    mean = pixel_sum / num_pixels
    std = np.sqrt((pixel_squared_sum / num_pixels) - np.square(mean))

    print(f"Mean: {mean}, Standard Deviation: {std}")

    return mean, std

def create_train_test_transformers(mean, std, random_rotation_degrees, random_affine_degrees, random_translation, brightness, contrast, saturation, hue):
    """ Create the train and test transformers.

    Args:
        mean (tuple): The mean of the input images.
        std (tuple): The standard deviation of the input images.
        random_rotation_degrees (int): The maximum rotation angle in degrees.
        random_affine_degrees (int): The maximum affine transformation angle in degrees.
        random_translation (tuple): The maximum translation in pixels.
        brightness (float): The brightness adjustment factor (range [0, ∞), where 0 means black).
        contrast (float): The contrast adjustment factor (range [0, ∞), where 0 means a solid gray image).
        saturation (float): The saturation adjustment factor (range [0, ∞), where 0 means grayscale).
        hue (float): The hue adjustment factor (range [-0.5, 0.5], where 0 means no change).

    Returns:
        tuple: A tuple containing two torchvision transform objects:
            - train_transform (torchvision.transforms.Compose): The transformation pipeline for training images.
            - test_transform (torchvision.transforms.Compose): The transformation pipeline for test images.
    """
    train_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(random_rotation_degrees),
        v2.RandomAffine(
            random_affine_degrees,
            translate=random_translation
        ),
        v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),
        v2.ToTensor(),
        v2.Normalize(mean=mean, std=std)
    ])

    test_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=mean, std=std)
    ])

    return train_transform, test_transform

def create_data_loaders(data_dir, batch_size, train_transform, test_transform, output_dir):
    """Create DataLoader objects for training and testing datasets and save them as pickle files.

    Args:
        data_dir (str): The directory containing the dataset.
        batch_size (int): The batch size for the DataLoader.
        train_transform (torchvision.transforms.Compose): The transformation pipeline for training images.
        test_transform (torchvision.transforms.Compose): The transformation pipeline for test images.
        output_dir (str): The directory to save the pickle files.
    """
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'train_loader.pkl'), 'wb') as f:
        pickle.dump(train_loader, f)

    with open(os.path.join(output_dir, 'test_loader.pkl'), 'wb') as f:
        pickle.dump(test_loader, f)

    print(f"Data loaders saved to {output_dir}")