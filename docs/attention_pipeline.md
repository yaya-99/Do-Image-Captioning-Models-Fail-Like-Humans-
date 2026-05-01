# Attention Extraction and Saliency Alignment Pipeline

This document describes the computer vision pipeline used to extract token-level cross-attention from image captioning models and align it with human saliency maps.

The purpose of this pipeline is **not to improve caption generation**, but to analyze whether model attention aligns with regions that humans consider visually salient.

---

## 1. Overview

The pipeline performs the following steps:

1. Generate captions using a pretrained image captioning model.
2. Extract cross-attention maps for each generated token.
3. Aggregate attention across heads and layers.
4. Resize attention maps to a common spatial resolution.
5. Align attention maps with human saliency maps.
6. Produce token-level attention distributions for divergence analysis.

This pipeline enables comparison between:

- **Model attention**
- **Human visual saliency**

---

## 2. Caption Generation

Images from the COCO validation dataset are passed through pretrained captioning models.

Models used in this project:

- **BLIP**
- **ViT-GPT2**

Each model generates a caption token-by-token.

For every token produced, the model provides **cross-attention weights** between:

- textual tokens
- spatial image features.

---

## 3. Cross-Attention Extraction

For each generated token:

1. Retrieve the cross-attention tensor from the model.
2. Extract attention weights corresponding to image feature tokens.
3. Aggregate attention values across attention heads.

The resulting output is a **spatial attention map** representing where the model attends while generating a specific token.

---

## 4. Spatial Resolution Alignment

Attention maps from transformer models are typically represented over image patches.

To ensure compatibility with human saliency maps:

- Attention maps are resized to a **24 × 24 grid**.
- Interpolation is applied where necessary.

This produces a standardized spatial representation of model attention.

---

## 5. Human Saliency Maps

Human visual attention is approximated using **SALICON saliency maps**.

Processing steps include:

1. Loading fixation-based saliency maps.
2. Normalizing saliency values.
3. Resizing maps to **24 × 24 resolution** to match model attention maps.
4. Converting maps into probability distributions.

---

## 6. Attention–Saliency Alignment

For each token:

- Model attention map  
- Human saliency map  

are compared after normalization.

Both maps are treated as probability distributions over spatial locations.

These aligned distributions allow divergence metrics to be computed.

---

## 7. Output for Analysis

The pipeline produces token-level attention distributions used in subsequent statistical analysis.

Outputs include:

- normalized attention maps
- normalized saliency maps
- token-level divergence metrics (JS and KL)

These outputs are stored in:

- results/tables/

and used in the statistical analysis notebooks.

---

## 8. Role in the Research Pipeline

This pipeline supports the broader research objective:

> determining whether attention alignment correlates with semantic correctness in generated captions.

Importantly:

- attention alignment is treated as **evidence of perceptual focus**
- semantic correctness is evaluated **separately through manual annotation**

This separation allows the study to investigate whether models exhibit **human-like perceptual attention while still producing incorrect semantic outputs.**

---

## 9. Implementation Location

The implementation of this pipeline is available in:

- analysis/vitgpt2_attention_analysis.ipynb

which demonstrates attention extraction and alignment for the ViT-GPT2 model.

