import yaml
import os
import shutil
import random
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def load_config(config_path):
    """ Load configuration from a YAML file

    Args:
        config_path (str): Path to the config YAML file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def split_data(input_dir, output_dir, train_ratio):
    """ Split the data into train and test sets

    Args:
        input_dir (str): Path to the input data directory
        output_dir (str): Path to the output data directory
        train_ratio (float): Ratio of the data to be used for training
    """
    # Check if input directory exists
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    # Check if the data has already been split
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print('Data already split.')
    else:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
    # Get the list of class folders
    class_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    

    for class_folder in tqdm(class_folders, desc="Iterating through object folders...", unit="folders"):
        class_path = os.path.join(input_dir, class_folder)

        # Get the list of images in the class folder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        # Split the images into train and test
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Create the class folders in train and test directories
        train_class_dir = os.path.join(train_dir, class_folder)
        test_class_dir = os.path.join(test_dir, class_folder)

        # Check if the class folders already exist
        if os.path.exists(train_class_dir) and os.path.exists(test_class_dir):
            print(f"Class {class_folder} already exists in train/test. Skipping.")
            continue
        
        # Create the class folders
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Copy the images to the class folders
        for img in tqdm(train_images, desc=f"Copying train images for {class_folder}", leave=False):
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

        for img in tqdm(test_images, desc=f"Copying test images for {class_folder}", leave=False):
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

    print("Data split completed successfully!")

def load_data_loader(pickle_path):
    """ Load data loader from a pickle file

    Args:
        pickle_path (str): Path to the pickle file

    Returns:
        object: Data loader object
    """
    with open(pickle_path, 'rb') as file:
        data_loader = pickle.load(file)
    return data_loader

def plot_metrics(train_values, test_values, metric_name):
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_values) + 1)

    sns.lineplot(x=epochs, y=train_values, label=f"Train {metric_name}")
    sns.lineplot(x=epochs, y=test_values, label=f"Test {metric_name}")

    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name}")
    plt.legend()
    plt.savefig(f"plots/{metric_name.lower()}_plot.png")
    plt.show()