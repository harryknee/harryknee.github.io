---
title: "DiscDiff: Latent Diffusion Model for DNA Sequence Generation"
date: "2023-09-28"
description: "Poster, AI4Science Workshop, NeurIPS 2023\n"
tags:
  - publication
  - ai4science
  - diffusion
  - VAE
---

**&commat; Authors** | *Zehui Li\*, <u>Yuhao Ni\*</u>, Tim August B. Huygelen, Akashaditya Das, Guoxuan Xia, Guy-Bart Stan, Yiren Zhao*

**&nearr; Full-text** | [https://arxiv.org/abs/2310.06150](https://arxiv.org/abs/2310.06150)

## Abstract

The harnessing of machine learning, especially deep generative models, has opened up promising avenues in the field of synthetic DNA sequence generation. Whilst Generative Adversarial Networks (GANs) have gained traction for this application, they often face issues such as limited sample diversity and mode collapse. On the other hand, Diffusion Models are a promising new class of generative models that are not burdened with these problems, enabling them to reach the state-of-the-art in domains such as image generation. In light of this, we propose a novel latent diffusion model, DiscDiff, tailored for discrete DNA sequence generation. By simply embedding discrete DNA sequences into a continuous latent space using an autoencoder, we are able to leverage the powerful generative abilities of continuous diffusion models for the generation of discrete data. Additionally, we introduce Fréchet Reconstruction Distance (FReD) as a new metric to measure the sample quality of DNA sequence generations. Our DiscDiff model demonstrates an ability to generate synthetic DNA sequences that align closely with real DNA in terms of Motif Distribution, Latent Embedding Distribution (FReD), and Chromatin Profiles. Additionally, we contribute a comprehensive cross-species dataset of 150K unique promoter-gene sequences from 15 species, enriching resources for future generative modelling in genomics.

## 