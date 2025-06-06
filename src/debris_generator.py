import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import List
from torchvision import transforms
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from segmentation_models_pytorch import Unet
from pathlib import Path 

class DebrisGenerator:
    def __init__(
        self,
        sd_model: str,
        controlnet_model: str,
        negative_prompt: str,
        bin_threshold: float
    ):
        """
        Initializes both the ControlNet-based generation pipeline and
        a U-Net segmentation model to extract masks from input images.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 

        # Load up the ControlNet model with half-precision to save GPU memory
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.float16
        )

        # Now wire that into a Stable Diffusion pipeline
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model,
            controlnet=controlnet,
            torch_dtype=torch.float16
        )

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        # memory-saving stuff
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_model_cpu_offload()

        self.pipe = pipeline.to(self.device)
        self.neg_prompt = negative_prompt
        self.bin_threshold = bin_threshold

        # Set up the segmentation model (U-Net + ResNet34 backbone)
        self.mask_net = Unet("resnet34", encoder_weights="imagenet", classes=1).to(self.device)
        self.mask_net.eval()  # don’t forget this, otherwise batchnorm goes wild

        # Basic preprocessing for segmentation input       
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_mask(self, image_rgb: Image.Image) -> Image.Image:
        """
        Feeds an RGB image through the U-Net to produce a binary mask.
        Returns a single-channel (L-mode) PIL image.
        """
        tensor_input = self.preprocess(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():  # no gradients needed for inference
            raw_output = self.mask_net(tensor_input)

        # Apply sigmoid and thresholding
        prob_map = torch.sigmoid(raw_output)[0, 0].cpu().numpy()
        binary_array = (prob_map > self.bin_threshold).astype(np.uint8) * 255

        # Convert NumPy array back into a grayscale image
        return Image.fromarray(binary_array, mode="L")

    def generate(
        self,
        input_images: List[str],
        prompts: List[str],
        output_folder: str,
        strength: float,
        guidance_scale: float,
        conditioning_scale: float,
        steps: int
    ) -> List[str]:
        """
        Main generation function:
        - Loads each image
        - Extracts a mask
        - Generates Canny edges
        - Feeds it all into ControlNet
        - Saves the resulting images
        """
        os.makedirs(output_folder, exist_ok=True)
        all_outputs = []

        for img_path in input_images:
            base_rgb = Image.open(img_path).convert("RGB").resize((512, 512))  # SD expects 512x512
            debris_mask = self.extract_mask(base_rgb)  # we'll ignore this mask for now

            # Generate a Canny edge map for ControlNet conditioning
            np_img = np.array(base_rgb)
            canny_edges = cv2.Canny(np_img, 100, 200)  # might tune these later
            edge_overlay = Image.fromarray(np.stack([canny_edges]*3, axis=-1))

            filename_root = Path(img_path).stem 

            for txt_prompt in prompts:
                # Run image generation with all the params and prompt
                result = self.pipe(
                    prompt=txt_prompt,
                    negative_prompt=self.neg_prompt,
                    image=edge_overlay,
                    controlnet_conditioning_scale=conditioning_scale,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps
                )

                for i, output_img in enumerate(result.images):
                    # Filename includes original image name and prompt (underscored)
                    safe_prompt = txt_prompt.replace(" ", "_")
                    output_filename = f"debris_{filename_root}_{safe_prompt}_{i}.png"
                    output_path = os.path.join(output_folder, output_filename)

                    output_img.save(output_path)
                    all_outputs.append(output_path)

        return all_outputs