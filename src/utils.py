import glob
from pathlib import Path
from typing import List

def list_images(dir_path: str, exts=("png", "jpg", "jpeg")) -> List[str]:
    """
    Return sorted list of image file paths in `dir_path`.
    """
    files = []
    for e in exts:
        files.extend(glob.glob(f"{dir_path}/*.{e}"))
    return sorted(files)

def ensure_dir(path: str):
    """
    Create `path` if it doesn’t exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)