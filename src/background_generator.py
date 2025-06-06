import os
import torch
from pathlib import Path  # Forgot to import this in the original :(
from PIL import Image
from typing import List
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from torchvision import transforms  # Note: not used, might remove later

class BackgroundGenerator:
    def __init__(self, model_identifier: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loading the Stable Diffusion img2img model with half precision
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_identifier,
            torch_dtype=torch.float16  # Seems to help reduce memory usage
        )

        # Switching out the default scheduler with a better one (usually helps improve results)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # two "magic lines"
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

        self.pipe = pipe.to(self.device)

    def generate(
        self,
        seed_paths: List[str],
        prompts: List[str],
        out_dir: str,
        strength: float = 0.75,  # setting a sensible default here just in case
        guidance_scale: float = 7.5,
        steps: int = 50
    ) -> List[str]:
        # Just making sure our output folder is there
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        output_files = []

        for img_path in seed_paths:
            try:
                # Load and prep the image – assuming everything is 512x512
                base_img = Image.open(img_path).convert("RGB")
                base_img = base_img.resize((512, 512))  # TODO: maybe make this resize optional?

                name_base = Path(img_path).stem  # Use filename as part of output name
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
                continue  # Skip to next image

            for prompt in prompts:
                # Running the actual image-to-image generation
                result = self.pipe(
                    prompt=prompt,
                    image=base_img,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps
                )

                # Save all images from result (usually just one, but just in case)
                for i, generated_img in enumerate(result.images):
                    safe_prompt = prompt.replace(" ", "_")[:40]  # Just to avoid long filenames
                    output_filename = f"{out_dir}/bg_{name_base}_{safe_prompt}_{i}.png"
                    generated_img.save(output_filename)
                    output_files.append(output_filename)

        return output_files
