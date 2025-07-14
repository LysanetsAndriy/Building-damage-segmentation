# Building-damage-segmentation

This repository contains the implementation and experiments for a building damage segmentation project based on a novel dataset of ground‐level images. The goal is to accurately identify and localize damaged parts of buildings, such as broken windows, damaged roofs, and general structural damage, using modern deep learning architectures.

# Project overview

**Dataset**

1. 290 high-resolution images of Ukrainian buildings captured from ground level.
2. Manual polygon annotations for up to six classes: Other, Building, Roof, Damage, Damaged Roof, Broken Window.
3. Severe class imbalance (e.g., “Other” ≈ 66%, “Roof” ≈ 1.2%), addressed via weighted loss functions and patch-level augmentation.

**Patch Generation**

* Each full image is sliced into fixed-size patches to increase data volume and fit GPU memory.
* Patch counts per image ranged from 18 to 44, depending on patch size and stride.
* Three patch configurations aligned with encoder input requirements:

  1. ResNet50: 224×224 patches (stride 180)
  2. SwinV2 Large: 384×384 patches (stride 180)
  3. ConvNeXt Large / SegFormer b5 / YOLO 11x-seg / DINOv2: 640×640 patches (stride 160)
 
**Model Architectures**

Base architecture: U-Net with standard encoder–decoder and skip connections.
Encoders evaluated:

* ResNet-50
* Swin Transformer V2 Large
* ConvNeXt Large
* Ultralytics YOLO 11x-seg
* DINOv2 ViT Large

Also, the SegFormer b5 model was evaluated separately from the U-Net architecture.

**Embeddings & Post-Processing**

Global embeddings: embeddings extracted from the original resized image via ConvNeXt-Large and concatenated at the U-Net bottleneck.

Positional embeddings: normalized (x,y) patch coordinates mapped to feature vectors and injected alongside global embeddings.

Superpixel smoothing: Felzenszwalb superpixels + majority voting refine patch-level predictions into coherent regions.

**Experiments**

24 distinct models:

* 6 encoders × 2 settings (with/without embeddings) × 2 post-processing options (with/without superpixels).

Training split: 85% of images → training patches; 15% → validation patches.

Loss: weighted sum of Cross-Entropy (0.3), IoU (0.35), Dice (0.35) with per-class weights to counter imbalance.

Optimizer: Adam + ReduceLROnPlateau on validation loss.

Metrics: per-class IoU, F1-score, and visual inspection of validation masks each epoch.

Best Results:

| Configuration                                   | Classes | Superpixels |   IoU  | F1-score |
| ----------------------------------------------- | :-----: | :---------: | :----: | :------: |
| **U-Net + DINOv2 encoder + embeddings**         |    6    |      No     | 0.4711 |  0.7462  |
| **U-Net + ResNet-50 encoder + embeddings + SP** |    3    |     Yes     | 0.7400 |  0.8714  |

* Six-class task: best achieved by DINOv2 backbone with global + positional embeddings.

* Three-class task (Other / Building / Damage): best achieved by ResNet-50 backbone with embeddings and superpixel smoothing.

**Notebooks**
All experiments are fully reproducible via the Colab notebooks in this repo, each corresponding to one encoder and one experimental setting (embeddings on/off):

**Each notebook includes:**

* Data loading and patch dataset creation

* Dataset and Dataloader creation

* Model architecture, training loop, and validation

* Model training

* Metric computation and mask visualizations

* Final model testing and metrics evaluation

