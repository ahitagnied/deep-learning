# deep-learning

A deep dive into Deep Learning, focusing on Computer Vision and Computational Imaging.  This repository progresses from implementing a custom Autograd engine in Python to training complex models like NeRFs, Gaussian Splatting, and GAN-Diffusion Hybrids on various datasets.  Each folder contains seminal papers and my implementations using open-source datasets. Jupyter Notebooks provide detailed explanations for each step, from model architecture to training.


## Table of Contents

- [Deep Neural Networks](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks)
  - [DNN](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/011-dnn):  Implementation of a custom Autograd engine and application to the Two Moons dataset.
  - [CNN](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/012-cnn): Convolutional Neural Network implementation and training on the MNIST dataset.
  - [LSTM](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/014-lstm): Long Short-Term Memory network for Amazon stock price forecasting.
  - [Alex-Net](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/013-alex-net): Implementation and training of AlexNet on the CelebA dataset.
  - [U-Net](https://github.com/ahitagnied/deep-learning/tree/main/01-deep-neural-networks/015-u-nets):  U-Net implementation and training on the Carvana dataset.
- [Optimization & Regularization](https://github.com/ahitagnied/deep-learning/tree/main/02-optimization-regularization/l1-l2)
  - L1/L2 Weight Decay
  - Activation Functions (ReLU, GELU, Swish, SELU)
  - Skip Connections & Residuals
  - Dropout, Stochastic Depth & ShakeDrop
  - Batch & Layer Normalization, Group Normalization
  - Optimizers (SGD, Adam, AdamW, AdaGrad, LookAhead)
  - Learning Rate Schedulers (CyclicLR, Warm Restarts)
  - Gradient Clipping & Gradient Accumulation
- [Sequence & Modelling](https://github.com/ahitagnied/deep-learning/tree/main/03-sequence-modelling)
  - Bigrams
  - MLP
  - RNN
  - GRU
  - Transformers
- [Vision & Representation](https://github.com/ahitagnied/deep-learning)
  - Attention Mechanisms (Scaled Dot-Product, Multi-Head Attention)
  - Vision Transformers (ViT, Swin, DeiT)
  - CLIP & Contrastive Learning
  - Self-Supervised Learning (BYOL, SimCLR, MoCo)
- [Advanced Vision](https://github.com/ahitagnied/deep-learning/tree/main/05-advanced-vision/051-diffusion-models)
  - Diffusion Models (DDPM, Stable Diffusion)
  - GANs 
  - NeRF
  - Implicit Neural Representations


## Features

- Implementations of various deep learning models from basic DNNs to advanced architectures like NeRFs and Diffusion Models.
- Jupyter Notebooks with detailed explanations and code for each model.
- Use of open-source datasets for training and evaluation.
- Exploration of different optimization and regularization techniques.
- Focus on computer vision and computational imaging applications.


## Usage

Each subdirectory within this repository contains a Jupyter Notebook demonstrating the implementation and training of a specific deep learning model. Follow the instructions within each notebook to run the code and reproduce the results.


## Installation

1. Clone the repository: `git clone https://github.com/ahitagnied/deep-learning.git`
2. Navigate to the project directory: `cd deep-learning`
3. Install the required Python packages.  A `requirements.txt` file will be made for the repository soon.


## Technologies Used

- **Python:** The primary programming language for all implementations.
- **PyTorch:** A popular deep learning framework used for building and training the models.
- **NumPy:** Used for numerical computations and array manipulation.
- **Pandas:** Used for data manipulation and analysis, especially in the time series forecasting example.
- **Matplotlib:** Used for visualizing data and results.
- **Scikit-learn:** Used for data preprocessing (MinMaxScaler) in the time series forecasting example.
- **TensorFlow:** Used in the Diffusion model example (note: other frameworks can be used).
- **Torchvision:** Used for loading and preprocessing image datasets (MNIST, CelebA, FGVCAircraft).
- **Graphviz:** Used for visualizing the computation graph in the backpropagation example.
- **tqdm:** Used for displaying progress bars during training in the DDPM example.


## Statistical Analysis

Statistical analysis methods employed vary depending on the specific project.  Examples include:

- **Negative Log-Likelihood:** Used as a loss function for evaluating model performance in the Bigram language model.
- **Mean Squared Error (MSE):** Used as a loss function for time series forecasting.
- **L1 Loss:** Used as a loss function in the Diffusion Model example.

Data preprocessing methods like normalization (MinMaxScaler) and one-hot encoding are used as needed.


## Dependencies

The necessary libraries can be installed using pip (a `requirements.txt` will be created soon).


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


## License

[MIT License](LICENSE)



*README.md was made with [Etchr](https://etchr.dev)*
