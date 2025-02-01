from src.transform import *
from src.utils import *
from model.resnet_18 import *

def main():
    config = load_config("config.yaml")
    mean, std = compute_mean_std("input", for_training=False)
    model = ResNet18()
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()

if __name__ == "__main__":
    main()