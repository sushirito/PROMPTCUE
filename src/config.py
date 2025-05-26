import yaml
from pathlib import Path

_config_path = Path(__file__).parent.parent / "config" / "config.yaml"
with open(_config_path, "r") as f:
    _cfg = yaml.safe_load(f)

DATA        = _cfg["data"]
MODELS      = _cfg["models"]
PROMPTS     = _cfg["prompts"]
IMG2IMG_OPS = _cfg["img2img"]
CN_OPS      = _cfg["controlnet"]
SEG_OPS     = _cfg["segmentation"]