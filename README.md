# Safe Diffusion Guidance — Demo (Classifier-Guided Sampling)

This repository contains a minimal demo of **Safe Diffusion Guidance**, an in-process safety mechanism for text-to-image generation.

---

## 🌐 Overview
Instead of blocking prompts or rejecting images *after* they are rendered, Safe Diffusion Guidance steers the **reverse-diffusion trajectory itself** toward safer outcomes while maintaining fidelity and alignment with the user prompt.  
The safety signal comes from a classifier over mid-UNet features, which guides the denoiser away from unsafe regions and toward “safe” features.

---

## ⚙️ Core Idea
- **Look-ahead (latent prediction) guidance**: estimate what the clean latent would look like if denoising stopped now, and apply a loss that discourages unsafe features.  
- **Classifier-based guidance**: gradients from a trained safety classifier attract latents toward *safe* features and repel unsafe ones.  

Together these modify the UNet score function, yielding denoiser-centric safety controls rather than surface-level filters.

---

## 🚀 What the Demo Does
The demo notebook will:
1. Load a **Stable Diffusion v1.5** pipeline (via 🤗 Diffusers).  
2. Download a **pre-trained safety classifier** from Hugging Face Hub.  
3. Generate two images per prompt:
   - **Original** (no safety guidance)  
   - **Safe (CG)** using classifier-guided sampling  
4. Run post-hoc classification on both to quantify the safety shift.  

---

## 🔧 Key Controls
- `NUM_STEPS` — number of denoising steps.  
- `CFG_SCALE` — classifier-free guidance scale (prompt fidelity).  
- `SAFETY_SCALE` — how strongly the safety guidance is applied.  
- `MID_FRACTION` — fraction of denoising steps where safety guidance is active.  
- `SAFE_IDX` — index of the “safe” class in classifier labels (`[gore, hate, medical, safe, sexual]`, here **safe = 3**).  

These correspond to the ablation axes in the paper.

---

## 📓 Usage
Open the provided Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/basimazam/safe_diffusion_demo/blob/main/Demo_Safe_Diffusion_Guidance.ipynb)

Follow the cells step by step:
- Enter your prompt (or select an example).
- Adjust `SAFETY_SCALE` and `MID_FRACTION` to explore the trade-off between image quality and safety.
- Compare Original vs Safe outputs side by side.

---

## 📂 Structure
safe_diffusion_demo/
│
├── Demo_Safe_Diffusion_Guidance.ipynb # Main Colab demo
├── adaptive_classifiers.py # Classifier utilities
├── custom_cg.py # Custom classifier guidance
└── README.md

---

## ⚠️ Disclaimer
This is a **demo only**.  
- The released classifier and settings are illustrative.  
- The implementation is research code; real deployments require **additional safeguards** and policy layers.  
- Detailed training code will be released upon acceptance of the associated paper.  

---