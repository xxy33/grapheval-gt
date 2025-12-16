import yaml
import torch
import random
import numpy as np
from typing import Dict, Any

def load_config(config_path: str = 'configs/config.yaml') -> Dict[str, Any]:
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False