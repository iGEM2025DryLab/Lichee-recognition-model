# Lichee Recognition Model Training Repository

Welcome to the Drylab's Lichee Recognition Model training repository. This repository contains all the necessary code and resources for training and using the lichee recognition models.

---

## Repository Overview

This repo is composed of two main parts:

### 1. `CNN.py`

- This Python script contains the core training program for the lichee recognition.
- It utilizes two primary convolutional neural network (CNN) architectures:
  - **ResNet-17**
  - **ResNet-34**
- After training, the models are saved (serialized) in the same folder as `CNN.py`.
- Each model requires a corresponding prediction file for inference.

### 2. Dataset

- The dataset used for training and testing the models is **not included** in this repository.
- Please download the dataset separately from your trusted source before starting training.

---

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch (or your preferred deep learning framework)
- Other dependencies as specified in `requirements.txt` (if available)

### Training

To train a model, run:

```bash
python CNN.py --model resnet-17
# or
python CNN.py --model resnet-34
