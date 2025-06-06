# PROMPTCUE

## Publication

View the original publication “Tackling Marine Pollution with IoT and Conditioned Diffusion” at the [2024 IEEE International Conference on Artificial Intelligence in Engineering and Technology (IICAIET)](https://ieeexplore.ieee.org/document/10730236).

---

## Overview

Marine plastic debris—particularly disposable masks—poses severe environmental and health risks. Traditional data collection is expensive, limited in scope, and slow to adapt to new pollutants. **PROMPTCUE** is a generative AI pipeline that addresses these challenges by rapidly synthesizing diverse, high-quality marine debris images using:

1. **Synthetic Background Generation**  
   - Applies Stable Diffusion Img2Img on a small set of seed marine scenes.  
   - Produces photorealistic “underwater,” “aerial,” and “surface” variants (e.g., turbidity, algal blooms, low-light conditions).

2. **Debris Aging & Biofouling**  
   - Uses a U-Net to segment mask shapes from seed images.  
   - Employs ControlNet (Canny conditioning + Stable Diffusion) with negative prompting to simulate biofouling, aging, and discoloration on masks.  

3. **Composite Overlay**  
   - Alpha-composites aged debris onto generated backgrounds to create realistic “marine pollution” scenarios.  

Within 24 GPU-hours on an NVIDIA A100, PromptCue generates **PROMPTCUE-Masks-60K**—a 60,000-image dataset featuring biofouled masks in varied marine environments. This dataset supports adaptive, real-time debris detection systems (e.g., PiDAR) and can be extended to other debris types or locations.

---

## Project Structure

```

PROPMTCUE/                                 ← Project root
├── config.yaml                            ← User-editable configuration
├── requirements.txt                       ← Python dependencies
├── setup.sh                               ← Installs all requirements
│
├── inputs/                                ← Seed images (download links below)
│   ├── backgrounds/                       ← Seed marine background images
│   └── masks/                             ← Seed mask images
│
├── outputs/                               ← All generated outputs
│   ├── backgrounds/                       ← SD Img2Img-generated backgrounds
│   ├── debris/                            ← ControlNet-aged debris images
│   └── composites/                        ← Final RGBA composites
│
└── src/                                   ← Source code
    ├── main.py                            ← Top-level orchestrator
    ├── config_loader.py                   ← Loads settings from config.yaml
    ├── utils.py                           ← Helpers: list\_images, ensure\_dir
    ├── background_generator.py            ← Img2Img background module
    ├── debris_generator.py                ← ControlNet debris module
    └── compositor.py                      ← Alpha composite logic

````

---

## Usage Instructions

### 1. Installation & Setup

1. **Clone this repository** and enter its root directory:
   ```bash
   git clone https://github.com/sushirito/PROMPTCUE.git
   cd PromptCue
   chmod +x setup.sh

2. **Run the setup script** to install Python dependencies:

   ```bash
   ./setup.sh
   ```

   This installs:

   * **diffusers**, **transformers**, **accelerate**
   * **segmentation-models-pytorch**, **opencv-python**, **xformers**, **torch**, **Pillow**, **PyYAML**

3. **Verify** that `config.yaml`, `src/`, and `requirements.txt` exist at the project root.

---

### 2. Preparing Input Data

PromptCue requires two small “seed” folders:

* **Seed Backgrounds**: Real-world marine scenes (underwater, aerial, surface).

  > [Download Seed Backgrounds from Google Drive](https://drive.google.com/drive/folders/1qNfyVFhfL07Tht1DtlWKrR2hyfxJETdC?usp=drive_link).
  > Once downloaded, place all images in:

  ```
  inputs/backgrounds/
  ```

* **Seed Masks**: Photographs of clean masks (surgical, N95, etc.) on a plain background.

  > [Download Seed Masks from Google Drive](https://drive.google.com/drive/folders/1rUfc2eQGrGtkPOHnL05OZRoP_uk3IvWC?usp=sharing).
  > After downloading, place them in:

  ```
  inputs/masks/
  ```

Ensure each directory contains only image files (`.png`, `.jpg`, `.jpeg`). No nested subfolders.

---

### 3. Configuring `config.yaml`

Open `config.yaml` at the project root to customize:

```yaml
data:
  backgrounds_dir: "inputs/backgrounds"
  masks_dir:       "inputs/masks"
  output_dir:      "outputs"

models:
  sd_model:         "runwayml/stable-diffusion-v1-5"
  controlnet_model: "lllyasviel/sd-controlnet-canny"

prompts:
  marine:
    - "ripples at dawn"
    - "turbid estuarine waters"
    - "dense algal blooms"
  debris:
    - "biofouled mask with barnacles and seaweed patches"
    - "sun-bleached mask coated in green-brown biofilm"
    - "algae-covered surgical mask floating in shallow water"
  negative: "clean, artificial, new, people, faces"

img2img:
  strength:       0.75
  guidance_scale: 7.5
  steps:          50

controlnet:
  strength:           0.5
  guidance_scale:     10.0
  conditioning_scale: 1.0
  steps:              50

segmentation:
  bin_threshold: 0.5
```

* `backgrounds_dir` and `masks_dir` point to your downloaded seed folders.
* `output_dir` is where all generated images will be saved.
* Modify `prompts`, `img2img`, `controlnet`, or `segmentation` parameters as desired.

---

### 4. Running the Pipeline

Once seeds are in place and `config.yaml` is configured, execute:

```bash
python -m src.main
```

This will run three sequential stages:

1. **Synthetic Background Generation**

   * Reads images from `inputs/backgrounds/`
   * Uses Stable Diffusion Img2Img to generate variants for each `prompts.marine` text.
   * Saves them under `outputs/backgrounds/`.

2. **Debris Aging & Biofouling**

   * Reads images from `inputs/masks/`
   * Segments mask shape with U-Net, applies Canny edges, then runs ControlNet with `prompts.debris` + `prompts.negative` to simulate biofouling.
   * Saves results under `outputs/debris/`.

3. **Composite Overlay**

   * Reads `outputs/backgrounds/` and `outputs/debris/`
   * Alpha-composites every debris image onto every background.
   * Saves final PNG composites under `outputs/composites/`.

At completion, you will see console logs like:

```
All outputs saved under 'outputs'
  • Synthetic backgrounds → outputs/backgrounds
  • Aged debris images    → outputs/debris
  • Final composites      → outputs/composites
```

---

## Directory Conventions

* **`inputs/`**: Raw seed images

  * `inputs/backgrounds/` → marine scenes
  * `inputs/masks/`       → clean mask photographs
* **`outputs/`**: Generated outputs

  * `outputs/backgrounds/` → Img2Img synthetic backgrounds
  * `outputs/debris/`      → ControlNet-aged debris images
  * `outputs/composites/`  → Final RGBA composites (background + debris)
* **`src/`**: Modular Python code

  * `main.py`              → Pipeline orchestrator
  * `config_loader.py`     → Loads settings from `config.yaml`
  * `utils.py`             → `list_images()`, `ensure_dir()` helpers
  * `background_generator.py` → Img2Img generation logic
  * `debris_generator.py`    → ControlNet debris aging logic
  * `compositor.py`          → Alpha composite implementation

---

## Citation

If you use **PROMPTCUE** in your work, please cite:

```bibtex
@inproceedings{shivakumar2024tackling,
  title={Tackling Marine Pollution with IoT and Conditioned Diffusion},
  author={Shivakumar, Aditya},
  booktitle={2024 IEEE International Conference on Artificial Intelligence in Engineering and Technology (IICAIET)},
  pages={142--146},
  year={2024},
  organization={IEEE}
}
```
