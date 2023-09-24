# Goal-Directed Molecule Generation using GCPN

This repository contains Python scripts demonstrating the application of Graph Convolutional Policy Network (GCPN) for generating molecules with specific desired properties. The primary focus is on maximizing the properties such as drug-likeness, represented by the QED score, and specific electronic energy types.

## Table of Contents
1. [Molecule Generation with GCPN - ZINC250k](#molecule-generation-with-gcpn---zinc250k)
2. [GCPN for Reinforcement Learning](#gcpn-for-reinforcement-learning)
3. [GCPN for QED - Quantitative Estimation of Drug-likeness](#gcpn-for-qed---quantitative-estimation-of-drug-likeness)
4. [Dependencies](#dependencies)
5. [Dataset](#dataset)
6. [Notes](#notes)

### Molecule Generation with GCPN - ZINC250k
[Link to Script](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/f56dff2773dd5b7deaf1e317966810d6de91cfe5/GNNs/molecue_generation/GCPN/pretraining_molecule_generation.ipynb)

This script showcases the use of GCPN on the ZINC250k dataset. The generated molecules are tuned to have properties represented by electronic energy types.

#### How to Run:
```bash
python pretraining_molecule_generation.py
```

### GCPN for Reinforcement Learning
[Link to Script](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/f56dff2773dd5b7deaf1e317966810d6de91cfe5/GNNs/molecue_generation/GCPN/plogp_based_molecule_generation.ipynb)

Property Specific Molecule Generation: GCPN for Reinforcement Learning. 
The script demonstrates molecule generation using GCPN with reinforcement learning. The dataset used is ZINC250k, and the primary goal is molecule generation optimized for properties described by the plogp measure.

How to Run:
```
python pretraining_molecule_generation.py
```

### GCPN for QED - Quantitative Estimation of Drug-likeness
[Link to Script](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/f56dff2773dd5b7deaf1e317966810d6de91cfe5/GNNs/molecue_generation/GCPN/qed_based_molecule_generation.ipynb)

In this script, GCPN is applied for generating molecules with a focus on maximizing their drug-likeness, as assessed by the QED score. The ZINC250k dataset is utilized, and the model aims to maximize the QED measure.

How to Run:
```bash
python qed_based_molecule_generation.py
```

### Dependencies
- [PyTorch](https://pytorch.org/)
- [TorchDrug](https://torchdrug.ai/docs/installation.html)

### Datasets
[ZINC250K](https://zinc.docking.org/)
---
# Notes
Make sure to replace `path_to_script_X.py` with the actual path to your Python scripts and `link_to_dataset` with a link to the ZINC250k dataset if you have one.
