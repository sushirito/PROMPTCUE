import glob
from pathlib import Path
from typing import List

def list_images(dir_path: str, exts=("png", "jpg", "jpeg")) -> List[str]:
    """
    Collects and returns a sorted list of image file paths from a given directory.
    Accepts common image extensions by default.
    """
    image_files = []

    for ext in exts:
        pattern = f"{dir_path}/*.{ext}"  # match files like *.png, *.jpg
        image_files.extend(glob.glob(pattern))

    return sorted(image_files)  # keeping things predictable

def ensure_dir(path: str):
    """
    Creates the specified directory if it doesn't exist yet.
    Handy for ensuring paths before saving stuff.
    """
    Path(path).mkdir(parents=True, exist_ok=True)  # recursive creation just in case