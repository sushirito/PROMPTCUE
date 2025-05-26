# promptcue/src/main.py

from config import DATA, MODELS, PROMPTS, IMG2IMG_OPS, CN_OPS, SEG_OPS
from utils import list_images, ensure_dir
from pipelines import Img2ImgGenerator, ControlNetGenerator

def main():
    out_dir = DATA["output_dir"]
    ensure_dir(out_dir)

    # 1. Synthetic backgrounds
    bg_seeds = list_images(DATA["backgrounds_dir"])
    img2img = Img2ImgGenerator(MODELS["sd_model"])
    bg_files = img2img.generate(
        seeds=bg_seeds,
        prompts=PROMPTS["marine"],
        out_dir=out_dir,
        strength=IMG2IMG_OPS["strength"],
        guidance=IMG2IMG_OPS["guidance_scale"],
        steps=IMG2IMG_OPS["steps"],
    )

    # 2. Debris aging
    mask_seeds = list_images(DATA["masks_dir"])
    cn = ControlNetGenerator(
        sd_model=MODELS["sd_model"],
        cn_model=MODELS["controlnet_model"],
        neg_prompt=PROMPTS["negative"],
        bin_thresh=SEG_OPS["bin_threshold"]
    )
    debris_files = cn.generate(
        seeds=mask_seeds,
        prompts=PROMPTS["debris"],
        out_dir=out_dir,
        strength=CN_OPS["strength"],
        guidance=CN_OPS["guidance_scale"],
        cond_scale=CN_OPS["conditioning_scale"],
        steps=CN_OPS["steps"],
    )

    # 3. Composite overlays
    img2img.composite(bg_files, debris_files, out_dir)

    print(f"All outputs saved to {out_dir}")

if __name__ == "__main__":
    main()
