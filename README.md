# Image JEPA Implementation

An implementation of a Joint Embedding Predictive Architecture (JEPA) for self-supervised learning on images.

## Overview

This repository contains a PyTorch implementation of JEPA for self-supervised visual representation learning. The architecture consists of:

- Context Encoder: A Vision Transformer that encodes visible patches of the input image
- Target Encoder: A Vision Transformer that encodes masked target regions
- Predictor: A transformer-based model that predicts target representations from context representations

## Features

- Patch-based image processing with configurable patch sizes
- Positional embeddings using sinusoidal encoding
- Multi-head self-attention mechanisms
- Context-target masking strategy for self-supervised learning
- Momentum-based target encoder updates
- Configurable model architecture (embedding dimensions, number of heads, layers etc.)

## Requirements

```
torch
torchvision
numpy
PIL
tqdm
matplotlib
```

## Model Architecture

### PatchEmbedder
- Divides images into non-overlapping patches
- Handles context and target patch sampling
- Removes overlapping patches between context and target regions

### PositionalEmbeddings
- Generates 1D and 2D sinusoidal positional embeddings
- Supports different grid sizes and embedding dimensions

### Vision Transformer
- Multi-head self-attention layers
- Feed-forward networks with residual connections
- Separate encoders for context and target regions

### Predictor
- Transformer-based architecture to predict target representations
- Uses masked positional embeddings

## Training

The model is trained using:
- Adam optimizer
- Mean squared error loss between predicted and target representations  
- Momentum update for target encoder parameters
- Configurable learning rate and batch size

## Usage

```python
# Initialize models
context_encoder = VisionTransformer(...)
target_encoder = VisionTransformer(...)
predictor = VisionTransformer_predictor(...)

# Train
train_iJEPA(context_encoder, target_encoder, predictor, num_epochs=10)
```

## Reference

Based on the JEPA architecture described in [JEPA paper/repository link]
