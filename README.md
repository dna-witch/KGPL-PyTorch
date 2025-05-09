# Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph
A PyTorch implementation of Knowledge Graph Pseudo-Labeling (KGPL), inspired by the paper:
*"Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph"*
by Riku Togashi, Mayu Otani, and Shin’ichi Satoh.
(https://arxiv.org/abs/2011.05061)

## Overview

This project provides a functional PyTorch implementation of the KGPL model, which addresses cold-start problems in personalized recommendation systems by pseudo-labeling over knowledge graphs. 

The recommendation model uses a knowledge graph to identify potential positive items for each user by focusing on neighbors in the graph structure, and treats unobserved user-item interactions as weakly-positive instances via pseudo-labeling. To mitigate popularity bias, the model uses an improved negative sampling strategy. The recommender also implements a co-training approach with dual student models to improve learning stability and robustness.

## Repository Structure

```
KGPL-PyTorch/
├── conf/                   # Configuration files for experiments
├── data/                   # Datasets and data loaders
├── preprocess/             # Data preprocessing scripts
├── utils/                  # Utility functions and evaluation metrics
├── model.py                # Implementation of the KGPL model
├── KGPL_MUSIC_FINAL_40.ipynb  # Example notebook demonstrating usage
├── requirements.txt        # Python dependencies
├── CHANGELOG.md            # Record of changes and updates
├── LICENSE                 # MIT License
└── README.md               # Project overview and instructions

```

## Getting Started

Run the following commands to clone the repository, create a virtual environment, and install the required packages to set up the model environment.

```bash
git clone https://github.com/dna-witch/KGPL-PyTorch.git
cd KGPL-PyTorch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage and Workflow
The `KGPL_MUSIC_FINAL_40.ipynb` notebook provides a step-by-step example of preprocessing data, co-training, and evaluating the KGPL model on a benchmark dataset. It's a great starting point to understand the workflow and experiment with the recommender model!


## Contributors
Shakuntala Mitra [@dna-witch](https://github.com/dna-witch/)

Taylor Hawks [@taylorhawks](https://github.com/taylorhawks/)


## Changelog
#### Adding to the changelog

First, identify the last commit hash recorded in `CHANGELOG.md`. Then, use the following command (replacing `LAST_COMMIT_HASH` with the actual hash):

> git log --pretty=format:"## %h%n #### %ad %n%n%s%n%n%b%n" --date=short LAST_COMMIT_HASH..HEAD >> CHANGELOG.md

This appends all new commits since `LAST_COMMIT_HASH` to the end of the changelog.

## Citation
If you find this implementation useful for your research, please cite the original paper:

```bibtex
@article{togashi2020alleviating,
  title={Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph},
  author={Togashi, Riku and Otani, Mayu and Satoh, Shin’ichi},
  journal={arXiv preprint arXiv:2011.05061},
  year={2020}
}
```

<!-- - Taylor notes 4/27 - 1030AM
  - I have a full pipeline working with datasets and dataloaders.  It trains and the loss goes down.
  - This is single learner, not colearning yet.
  - Fixed a bug where the training set would contain data without positive examples for one or more users.
  - Need to refactor dataset slightly - it's a bit hard to understand still.
  - I haven't touched the "aggregate", "get_neighbors", or aggregator objects yet, I only used the basic Aggregator that already was in the code.  Need help with this.
  - Also validation won't work yet since I need to clean up the datasets/dataloaders a bit of a refactor still.
  - Haven't gotten to evaluation yet at all either. -->
