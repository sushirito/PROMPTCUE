import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import List
from torchvision import transforms
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from segmentation_models_pytorch import Unet

class Img2ImgGenerator:
    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        self.pipe = pipe.to(self.device)

    def generate(
        self,
        seeds: List[str],
        prompts: List[str],
        out_dir: str,
        strength: float,
        guidance: float,
        steps: int
    ) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        outputs = []
        for pth in seeds:
            init_img = Image.open(pth).convert("RGB").resize((512,512))
            base = os.path.splitext(os.path.basename(pth))[0]
            for prm in prompts:
                res = self.pipe(
                    prompt=prm,
                    image=init_img,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=steps
                )
                for i, img in enumerate(res.images):
                    fn = f"{out_dir}/bg_{base}_{prm.replace(' ','_')}_{i}.png"
                    img.save(fn)
                    outputs.append(fn)
        return outputs

    def composite(self, bgs: List[str], debris: List[str], out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        for bg_fp in bgs:
            bg = Image.open(bg_fp).convert("RGBA")
            base = os.path.splitext(os.path.basename(bg_fp))[0]
            for d_fp in debris:
                dm = Image.open(d_fp).convert("RGBA").resize(bg.size)
                comp = Image.alpha_composite(bg, dm)
                out_fn = f"{out_dir}/composite_{base}_{os.path.basename(d_fp)}"
                comp.save(out_fn)

class ControlNetGenerator:
    def __init__(
        self,
        sd_model: str,
        cn_model: str,
        neg_prompt: str,
        bin_thresh: float
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cn = ControlNetModel.from_pretrained(cn_model, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model, controlnet=cn, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        self.pipe = pipe.to(self.device)

        self.neg_prompt = neg_prompt
        self.bin_thresh = bin_thresh

        # segmentation U-Net
        self.seg = Unet("resnet34", "imagenet", classes=1).to(self.device)
        self.seg.eval()
        self.proc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

    def extract_mask(self, img: Image.Image) -> Image.Image:
        x = self.proc(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.seg(x)
        m = torch.sigmoid(pred)[0,0].cpu().numpy()
        bin_m = (m > self.bin_thresh).astype(np.uint8) * 255
        return Image.fromarray(bin_m, mode="L")

    def generate(
        self,
        seeds: List[str],
        prompts: List[str],
        out_dir: str,
        strength: float,
        guidance: float,
        cond_scale: float,
        steps: int
    ) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        outputs = []
        for pth in seeds:
            src = Image.open(pth).convert("RGB").resize((512,512))
            mask = self.extract_mask(src)
            arr = np.array(src)
            edges = cv2.Canny(arr, 100, 200)
            cond_img = Image.fromarray(np.stack([edges]*3, axis=-1))
            base = os.path.splitext(os.path.basename(pth))[0]
            for prm in prompts:
                res = self.pipe(
                    prompt=prm,
                    negative_prompt=self.neg_prompt,
                    image=cond_img,
                    controlnet_conditioning_scale=cond_scale,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=steps
                )
                for i, img in enumerate(res.images):
                    fn = f"{out_dir}/debris_{base}_{prm.replace(' ','_')}_{i}.png"
                    img.save(fn)
                    outputs.append(fn)
        return outputs
