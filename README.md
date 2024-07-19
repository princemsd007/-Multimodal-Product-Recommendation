# Multi-Modal Recommendation System

## Introduction

This repository contains the implementation of a Multi-Modal Attention based Siamese Network for unsupervised video rating prediction, as described in the paper "Towards Developing a Multi-Modal Video Recommendation System".

Our system aims to address the cold start and data sparsity problems in OTT platform recommendation systems by leveraging multi-modal information alongside rating data.

## Features

- Utilizes text, image, and metadata for comprehensive product representation
- Implements a Siamese network architecture for learning product similarities
- Supports both training from scratch and inference using pre-trained models

## Dataset

We use an enhanced version of the MovieLens 100K dataset, enriched with:
- Textual summaries from IMDb
- Video trailers from YouTube
- Metadata (Directors, Cast, Rating, Duration, etc.)

Pre-computed embeddings for each modality are provided in `Datasets/ml-100k/<modality>/embeddings.csv`.

## Requirements

- Python 3.10.9
- PyTorch 1.13.1
- CUDA 11.7 (for GPU acceleration)
- Additional libraries: torchsampler, numpy, pandas, tqdm, sklearn

## Installation

1. Clone this repository:git clone https://github.com/princemsd007/-Multimodal-Product-Recommendation.git
2. Install the required dependencies: pip install -r requirements.txt

   
## Usage

1. Ensure all `embeddings.csv` files are in place for each modality.
2. Open and run all cells in `Experiments/Siamese_Network.ipynb`.
3. Adjust epochs and hyperparameters as needed.
4. For inference only, skip the training module and run cells under the test module.

## Training on Custom Dataset

1. Access the enhanced dataset [here]([link-to-dataset](https://grouplens.org/datasets/movielens/100k/)).
2. Generate embeddings using scripts in `Experiments/embedding_generation/`.
3. Store embeddings in respective `Datasets/ml-100k/<modality>/embeddings.csv` files.
4. Ensure CSV files (1682 x N) maintain the same order as `items.csv` (1682 x M).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


