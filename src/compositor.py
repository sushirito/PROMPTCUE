import os
from pathlib import Path 
from PIL import Image
from typing import List

def composite_images(
    background_paths: List[str],
    debris_paths: List[str],
    out_dir: str
) -> None:
    """
    Loops through all backgrounds and debris layers,
    resizes debris to match each background, and composites them together.
    Saves everything into the specified output folder.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for bg_file in background_paths:
        try:
            bg_image = Image.open(bg_file).convert("RGBA")
        except Exception as e:
            print(f"Error loading background image {bg_file}: {e}")
            continue

        bg_name = Path(bg_file).stem

        for debris_file in debris_paths:
            try:
                debris_img = Image.open(debris_file).convert("RGBA")
            except Exception as e:
                print(f"Error loading debris image {debris_file}: {e}")
                continue

            debris_img = debris_img.resize(bg_image.size)

            # Composite using alpha channel, assumes both images are RGBA
            combined = Image.alpha_composite(bg_image, debris_img)

            # Build output filename; might be worth truncating names to avoid filesystem issues
            debris_name = Path(debris_file).name  # keeping extension here for clarity
            out_filename = f"composite_{bg_name}_{debris_name}"
            out_path = os.path.join(out_dir, out_filename)

            # Save result but might consider converting back to RGB if we don’t need transparency
            combined.save(out_path)

            # Optional: could add a print to track what's being saved
            # print(f"Saved: {out_path}")