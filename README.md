# Image-GS Implementation

[![Status: Learning Project](https://img.shields.io/badge/status-learning-blue)](#)

Minimal, educational PyTorch implementation of the paper “Image-GS: Content-Adaptive Image Representation via 2D Gaussians.”

- Paper: [Image-GS: Content-Adaptive Image Representation via 2D Gaussians](https://arxiv.org/pdf/2407.01866)

## Status

This is a personal learning project. It is incomplete, unoptimized, and code may change.

## Overview

This repo explores the core idea of Image-GS: representing an image with a set of 2D Gaussians. The goal is to keep the math and code minimal while demonstrating how to optimize Gaussian parameters to reconstruct a target image using PyTorch.

## How it works (analogy)

- Traditional heightmap/FdF-style projects: project known points (e.g., via isometric projection) and draw them.
- Here: initialize 2D Gaussians randomly and use PyTorch optimization to fit their parameters so that the rendered Gaussians approximate the input image.

## Quick start

Requirements:
- Python 3.x
- PyTorch
- numpy, pillow, matplotlib

Run:
- Edit the image path in [`Minimal2dGaussian.py`](https://github.com/fgarcia42/Image-Gs_Implementation/blob/c3f403085cef64cd000b7ef80f1f454ce9814e03/Minimal2dGaussian.py) (variable `path` near the bottom).
- Then run:
  ```bash
  python Minimal2dGaussian.py
  ```

## Roadmap / TODO

- Implement additional components from the paper
- Improve optimization stability and performance
- Add CLI, examples, and evaluation metrics
- Provide demo images/GIFs and benchmarks
- Write tests and documentation
