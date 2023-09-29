# Playground for experimenting with diffusion models 🌀

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository includes the following:
- `diffusion` package that provides a clean, modular, and minimalistic implementation of different components and algorithms used in diffusion-based generative modeling (with references to key papers), and
- `playground` folder that contains a collection of concrete examples that demonstrate diffusion-based generative modeling on different kinds of data (2D points, MNIST, CIFAR10, 3D point clouds, etc.)


## Diffusion

TODO: describe package structure


## Playground

### 1. 2D points diffusion

This is a very toy example, where each data instance is a 2D point that lies on a swiss-roll 1D manifold.
Given that the data is so simple, it's a perfect playground for experimenting with different approaches to training and inference, visualizing diffusion trajectories, and building intuition.
Both training and inference can comfortably run on a laptop (it takes a minute or so to train the model to convergence).

<p align="center"><img src="./assets/points_2d-diffusion.gif" width="700px" /></p>

Colab notebook: (TODO: add link to the notebook)

### 2. MNIST diffusion

Another toy example, where diffusion model is trained on MNIST.
Model architectures are scaled down versions of the U-nets used on CIFAR10 and ImageNet benchmarks (all the architecture code is copied from https://github.com/NVlabs/edm/blob/main/training/networks.py verbatim).
It takes about 1 hour to train an MNIST denoiser in Google Colab using a T4 GPU for 20 epochs or so.
And running inference takes just a few seconds.

<p align="center"><img src="./assets/mnist-diffusion.gif" width="700px" /></p>

Colab notebook: (TODO: add link to the notebook)
