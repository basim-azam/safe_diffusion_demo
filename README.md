# Safe Diffusion Guidance — Demo

This repository provides a minimal Colab demo for:

**Principled Latent Trajectory Guidance and Benchmarking for Safer Text-to-Image Generation**  
_Basim Azam, Naveed Akhtar, Muzammal Naseer, Salman Khan, Mubarak Shah_

---

## ⚠️ Content Warning
Generated images **may include nudity or other sensitive content**.

- The **first 4 examples** are from figures shown in the associated paper.  
- Use the provided controls (`Prompt`, `Seed`, `Safety Scale`, `Mid Fraction`) to explore safe vs. original generations.  
- A `Blur sensitive` option is included to automatically blur outputs flagged as unsafe.  

This demo is released **for research purposes only** and is **not a production safety filter**.

---

## 🚀 Run the Demo
Open in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/basim-azam/safe_diffusion_demo/blob/main/Demo_Safe_Diffusion_Guidance.ipynb)

### Steps
1. **Open the notebook in Colab** (link above).  
2. Run the setup cells to install dependencies and load the models.  
3. Choose an **Example** or enter your own **Prompt**.  
4. Adjust parameters:
   - `Seed` (reproducibility, or randomize)
   - `Safety Scale` (higher → stronger safety guidance)
   - `Mid Fraction` (portion of denoising steps guided)
5. Tick the **Consent / Blur sensitive** options as needed.  
6. Click **Generate** to see **Original vs Safe** outputs side by side, along with classifier readouts.  

---

## 📂 Structure

---
safe_diffusion_demo/
├── Demo_Safe_Diffusion_Guidance.ipynb # Main Colab demo
├── adaptive_classifiers.py # Classifier utilities
├── custom_cg.py # Custom classifier guidance
└── README.md

---

## ⚠️ Disclaimer
- This is **research code** only.  
- The released classifier and settings are illustrative.  
- Deployments require **additional safeguards, policy enforcement, and moderation layers**.
