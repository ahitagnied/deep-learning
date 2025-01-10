## deep-learning
---

A deep dive on Deep Learning, with a focus in Computer Vision and Computational Imaging. I start with implementing an Autograd Engine in Python, and slowly build up from basic DNNs to train toy datasets such as TwoMoons, CNNs, Alex-Nets, & U-Nets to more complex CV and CL models such as NeRFs, Gaussian Splatting, and GAN-Diffusion Hybrids.

In each folder I have seminal papers related to the folder, and my own implementation of the models using an open source dataset. The Jupyter Notebooks containing my implementations have explanations of every step from Model Architecture to Training. 

## Table of Contents
---

- [Deep Neural Networks](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks)
  - [DNN](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/011-dnn): Autograd Engine using Python, TwoMoons 
  - [CNN](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/012-cnn): MNIST Dataset
  - [LSTM](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/014-lstm): Amazon Stock Forecasting
  - [Alex-Net](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/013-alex-net): CelebA Dataset
  - [U-Net](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/015-u-nets): Carvana Classification Challenge Dataset
- Optimization & Regularization
  - Weight Decay & L1/L2
  - Activation Functions (ReLU, GELU, Swish, SELU)
  - Skip Connections & Residuals
  - Dropout, Stochastic Depth & ShakeDrop
  - Batch & Layer Normalization, Group Normalization
  - Optimizers (SGD, Adam, AdamW, AdaGrad, LookAhead)
  - Learning Rate Schedulers (CyclicLR, Warm Restarts)
  - Gradient Clipping & Gradient Accumulation
- Vision & Representation
  - Attention Mechanisms (Scaled Dot-Product, Multi-Head Attention)
  - Vision Transformers (ViT, Swin, DeiT)
  - CLIP & Contrastive Learning
  - Self-Supervised Learning (BYOL, SimCLR, MoCo)
  - Dynamic Neural Networks
  - Cross-Modal Learning
- Advanced Vision
  - Diffusion Models (DDPM, Stable Diffusion)
  - GANs (CycleGAN, StyleGAN, BigGAN) & VAEs (β-VAE, Conditional VAE)
  - Neural Fields & Implicit Representations
  - NeRF Architecture & Variants (Mip-NeRF, NeRF-W)
  - Instant-NGP
  - Point Clouds & Mesh Reconstruction (PointNet, PointNet++)
  - Implicit Neural Representations
- Neural Rendering
  - Volume Rendering
  - Ray Marching
  - Multi-View Consistency
  - View Synthesis
  - Neural Radiance Caching
  - Real-Time Neural Graphics
  - Mesh-based Rendering
  - Hybrid Neural Rendering Techniques
- Emerging Areas
  - Diffusion-GAN Hybrids
  - Reinforcement Learning in Vision (e.g., visual navigation)
  - Federated Learning in Vision
  - Edge and Mobile AI (e.g., EdgeNeRF, MobileNetV3)
  - Quantum Neural Networks for Vision
