import yaml
from pathlib import Path

# Attempting to locate the config file one level up from this script
config_path = Path(__file__).resolve().parent.parent / "config.yaml"

# Read and parse the YAML config file
try:
    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
except FileNotFoundError:
    raise RuntimeError(f"Couldn't find config.yaml at expected path: {config_path}")
except yaml.YAMLError as e:
    raise RuntimeError(f"YAML parsing error: {e}")

# Unpacking top-level sections for easier access throughout the project
DATA        = config_data.get("data", {})
MODELS      = config_data.get("models", {})
PROMPTS     = config_data.get("prompts", {})
IMG2IMG_OPS = config_data.get("img2img", {})
CN_OPS      = config_data.get("controlnet", {})
SEG_OPS     = config_data.get("segmentation", {})

# Note: using .get() with default {} to avoid KeyErrors if a section is missing