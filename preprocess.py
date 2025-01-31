from src.utils import *
from src.transform import *

import warnings
warnings.filterwarnings("ignore")

config = load_config("config.yaml")

def preprocess():
    split_data(config["input_dir"], config["output_dir"], config["train_ratio"])
    mean, std = compute_mean_std(config["train_dir"])
    train_transform, test_transform = create_train_test_transformers(
        mean,
        std,
        config["random_rotation_degrees"],
        config["random_affine_degrees"],
        config["random_translation"],
        config["brightness"],
        config["contrast"],
        config["saturation"],
        config["hue"]
    )
    create_data_loaders("split_data", config["batch_size"], train_transform, test_transform, config["loader_dir"])

if __name__ == "__main__":
    preprocess()