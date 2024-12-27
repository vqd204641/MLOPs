import os
import yaml

with open(os.path.join("..", "config.yaml"), 'r') as file:
    config = yaml.safe_load(file)

DATA_CONFIG = config["data"]
MODEL_CONFIG = config["model"]
LOGGING_CONFIG = config["logging"]
