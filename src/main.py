from pathlib import Path
from promptcue.config_loader import DATA, MODELS, PROMPTS, IMG2IMG_OPS, CN_OPS, SEG_OPS
from promptcue.utils import list_images, ensure_dir
from promptcue.background_generator import BackgroundGenerator
from promptcue.debris_generator import DebrisGenerator
from promptcue.compositor import composite_images

def main():
    # --- Load paths from the config file ---
    bg_seed_folder = DATA["backgrounds_dir"]
    debris_seed_folder = DATA["masks_dir"]
    output_base_dir = DATA["output_dir"]

    ensure_dir(output_base_dir)  # just in case it doesn't exist yet

    # === Step 1: Generate backgrounds ===
    bg_output = Path(output_base_dir) / "backgrounds"
    ensure_dir(str(bg_output))

    seed_bg_imgs = list_images(bg_seed_folder)
    bg_generator = BackgroundGenerator(MODELS["sd_model"])

    print(f"[1/3] Generating synthetic backgrounds from {len(seed_bg_imgs)} seeds...")

    bg_files = bg_generator.generate(
        seed_paths=seed_bg_imgs,
        prompts=PROMPTS["marine"],
        out_dir=str(bg_output),
        strength=IMG2IMG_OPS["strength"],
        guidance_scale=IMG2IMG_OPS["guidance_scale"],
        steps=IMG2IMG_OPS["steps"],
    )

    # === Step 2: Generate aged debris ===
    debris_output = Path(output_base_dir) / "debris"
    ensure_dir(str(debris_output))

    debris_seeds = list_images(debris_seed_folder)
    debris_generator = DebrisGenerator(
        sd_model=MODELS["sd_model"],
        controlnet_model=MODELS["controlnet_model"],
        negative_prompt=PROMPTS["negative"],
        bin_threshold=SEG_OPS["bin_threshold"]
    )

    print(f"[2/3] Aging debris masks using ControlNet...")

    debris_files = debris_generator.generate(
        seed_paths=debris_seeds,
        prompts=PROMPTS["debris"],
        out_dir=str(debris_output),
        strength=CN_OPS["strength"],
        guidance_scale=CN_OPS["guidance_scale"],
        conditioning_scale=CN_OPS["conditioning_scale"],
        steps=CN_OPS["steps"],
    )

    # === Step 3: Composite backgrounds with debris ===
    composite_output = Path(output_base_dir) / "composites"
    ensure_dir(str(composite_output))

    print(f"[3/3] Compositing debris onto backgrounds...")

    composite_images(
        background_paths=bg_files,
        debris_paths=debris_files,
        out_dir=str(composite_output)
    )

    # Final report
    print(f"\n All done :-) Outputs are saved in '{output_base_dir}'")
    print(f"   - Backgrounds: {bg_output}")
    print(f"   - Debris:      {debris_output}")
    print(f"   - Composites:  {composite_output}\n")

if __name__ == "__main__":
    main()