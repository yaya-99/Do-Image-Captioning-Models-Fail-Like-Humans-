# Hallucination Analysis in Image Captioning Models

## Overview

Hallucination is a well-documented failure mode in image captioning systems where a model generates objects, attributes, or relationships that are **not present in the image**.  

Unlike simple recognition errors, hallucinations often arise from **strong language priors learned during training**, causing models to produce plausible but incorrect descriptions.

This document describes how hallucination errors were identified and analyzed in this project.

---

## Role in the Research Project

This analysis supports the broader research question:

**RQ2 — Hallucination & Prior Bias**

> Are hallucination errors driven by contextual language priors rather than perceptual evidence?

Understanding hallucinations is important because it helps determine whether captioning models behave more like:

- **perceptual systems** (grounded in visual evidence), or  
- **language systems** (driven by statistical co-occurrence patterns).

---

## Definition of Hallucination

In this project, a hallucination occurs when a captioning model generates:

- an **object**
- an **attribute**
- or a **relation**

that **cannot be visually verified in the image**.

Examples include:

| Generated Caption | Image Content | Error |
|-------------------|--------------|------|
| "A man riding a horse on the beach." | No horse visible | Object hallucination |
| "A red car parked on the street." | Car present but not red | Attribute hallucination |
| "A dog sitting on a couch." | Dog present but not on couch | Relation hallucination |

---

## Types of Hallucination

Hallucination errors can arise in several ways.

### 1. Object Hallucination

The model introduces an object that does not exist in the image.

Example:

- Caption: "A person holding a tennis racket."
- Reality: No tennis racket is visible.

Possible cause:
- strong training correlation between sports scenes and tennis equipment.

---

### 2. Attribute Hallucination

The model assigns an incorrect attribute to a visible object.

Example:

- Caption: "A red bus driving down the road."
- Reality: The bus is blue.

Possible cause:
- learned biases about common object appearances.

---

### 3. Relation Hallucination

The model predicts an incorrect spatial or semantic relationship.

Example:

- Caption: "A cat sitting on a table."
- Reality: The cat is on the floor.

Possible cause:
- frequent co-occurrence patterns in training captions.

---

## Annotation Procedure

Hallucination detection was performed through **manual inspection of generated captions**.

The process included:

1. Generate captions for a subset of images.
2. Compare generated captions with the original image.
3. Identify tokens corresponding to incorrect objects, attributes, or relations.
4. Label the error category.

Each caption was annotated with:

- **error presence** (binary)
- **error type**
- **primary error token**

These annotations are stored in:

- data/annotations/manual_labels.csv

---

## Hallucination and Language Priors

Image captioning models are trained on large datasets of image–caption pairs.  
As a result, they learn strong statistical associations such as:

- *people → holding objects*
- *dogs → sitting on couches*
- *plates → containing food*

These associations can cause the model to generate plausible but incorrect captions.

This behavior suggests that hallucinations may be driven by **language priors rather than visual evidence**.

---

## Relation to Attention Alignment

One of the key hypotheses of this project is that hallucinations may occur **even when the model attends to visually relevant regions**.

This means:

- the model may focus on the correct part of the image
- but still generate an incorrect token due to language bias.

Testing this hypothesis requires comparing:

- **model attention maps**
- **human saliency maps**

and analyzing divergence between them.

---

## Implications for Cognitive Analysis

From a cognitive perspective, hallucination errors resemble **prior-driven perceptual inference** in humans.

Humans sometimes misinterpret ambiguous visual scenes when strong expectations influence perception.

Similarly, captioning models may rely on:

- learned contextual expectations
- statistical regularities in language

instead of purely visual evidence.

This makes hallucination an important phenomenon for studying **AI perception errors through a cognitive lens**.

---

## Current Observations

Preliminary analysis of generated captions suggests that hallucinations occur in multiple contexts, including:

- scenes involving common household objects
- sports-related imagery
- scenes with animals or people

Further analysis is required to determine:

- whether hallucinations correlate with attention misalignment
- whether specific object categories are more prone to hallucination
- how hallucinations compare with other error types.

---

## Next Steps

Future analysis will focus on:

- quantifying hallucination frequency across models
- analyzing token-level attention patterns during hallucination
- comparing divergence metrics between hallucinated tokens and correct tokens.

These steps will help determine whether hallucinations arise from **visual misperception or language-driven inference**.