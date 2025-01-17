from pathlib import Path
import os
from typing import Dict, Any
import yaml


def load_od_config() -> Dict[str, Any]:
    # Load the existing YAML config
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'object_detection_config.yml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # add any additional configuration keys here
    # config[some_calculated_property] = value
    config['CATEGORIES'] = ['object' for _ in range(config['CLASSES'])]

    return config


CONFIG = load_od_config()

dataset_path = str(Path(CONFIG['dataset_path']).absolute())
