I just started learning about application of generative graph neural nets and inverse design with AI. I used PyTorch and TorchDrug python libraries and their tutorials to help me understand what is generative GNNs are and how they can be used for generating molecules. This repository contains Python scripts demonstrating the application of Graph Convolutional Policy Network (GCPN) for generating molecules with specific desired properties. The primary focus is on maximizing the properties such as drug-likeness, represented by the QED score, and specific electronic energy types.

# Goal-Directed Molecule Generation using Generative Model GCPN :pill:

## :bookmark_tabs: Table of Contents
**Generative Model GCPN for Molecule Generation**
1. [Molecule Generation with GCPN - ZINC250k](#molecule-generation-with-gcpn---zinc250k)
2. [GCPN for pLOGP - Solubility Driven Drug Generation](#gcpn-for-pLOGP)
3. [GCPN for QED - Quantitative Estimation of Drug-likeness](#gcpn-for-qed---quantitative-estimation-of-drug-likeness)

**Generative Model GraphAF for Molecule Generation**
1. [Molecule Generation with GraphAF - ZINC250k](#molecule-generation-with-graphAF---zinc250k)
2. [GraphAF for pLOGP - Solubility Driven Drug Generation](#graphaF-for-plogp-solubility-driven-drug-generation)
3. [GraphAF for QED - Quantitative Estimation of Drug-likeness](#graphAF-for-qed---quantitative-estimation-of-drug-likeness)

- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Notes](#notes)

### Molecule Generation with GCPN - ZINC250k
[Link to Script](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/f56dff2773dd5b7deaf1e317966810d6de91cfe5/GNNs/molecue_generation/GCPN/pretraining_molecule_generation.ipynb)

This script showcases the use of GCPN on the ZINC250k dataset. The generated molecules are tuned to have properties represented by electronic energy types.

#### How to Run :runner::
```bash
python pretraining_molecule_generation.py
```

### GCPN for pLOGP
[Link to Script](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/f56dff2773dd5b7deaf1e317966810d6de91cfe5/GNNs/molecue_generation/GCPN/plogp_based_molecule_generation.ipynb)

Property Specific Molecule Generation: GCPN for Reinforcement Learning. 
The script demonstrates molecule generation using GCPN with reinforcement learning. The dataset used is ZINC250k, and the primary goal is molecule generation optimized for properties described by the plogp measure.

How to Run :runner::
```
python pretraining_molecule_generation.py
```

### GCPN for QED - Quantitative Estimation of Drug-likeness
[Link to Script](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/f56dff2773dd5b7deaf1e317966810d6de91cfe5/GNNs/molecue_generation/GCPN/qed_based_molecule_generation.ipynb)

In this script, GCPN is applied for generating molecules with a focus on maximizing their drug-likeness, as assessed by the QED score. The ZINC250k dataset is utilized, and the model aims to maximize the QED measure.

How to Run :runner::
```bash
python qed_based_molecule_generation.py
```

### Molecule Generation with GraphAF - ZINC250k
[Link to Script](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/83faa27e5ed748b316c6d3789555231c7822c866/GNNs/molecue_generation/GraphAF/pretrained_molecule_generation.ipynb)

This script utilizes the Graph Autoregressive Flow (GraphAF) model to generate molecules from the ZINC250k dataset without any specific optimization goal. After defining the model and training parameters, the script trains the model and then generates molecular structures, displaying them in SMILES format.

### GraphAF for pLOGP - Solubility Driven Drug Generation
[Link to Script](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/83faa27e5ed748b316c6d3789555231c7822c866/GNNs/molecue_generation/GraphAF/plogp_based_molecule_generation.ipynb)

Employing the Graph Autoregressive Flow (GraphAF) model, this script aims to generate molecules optimized for the pLOGP property, a measure of lipophilicity. It loads a pre-trained model, fine-tunes it using reinforcement learning targeting the pLOGP property, and then generates optimized molecular structures, displaying them in SMILES format.

### GraphAF for QED - Quantitative Estimation of Drug-likeness
[Link to Script](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/83faa27e5ed748b316c6d3789555231c7822c866/GNNs/molecue_generation/GraphAF/qed_based_molecule_generation.ipynb)

This script uses the Graph Autoregressive Flow (GraphAF) model to create molecules with a focus on maximizing the Quantitative Estimation of Drug-likeness (QED) score. It loads a pre-trained model, refines it through reinforcement learning to enhance the QED score, and then constructs molecules optimized for drug-likeness, presenting them in SMILES notation.

### :wrench: Dependencies
- [PyTorch](https://pytorch.org/)
- [TorchDrug](https://torchdrug.ai/docs/installation.html)

### :books: Datasets
[ZINC250K](https://zinc.docking.org/)
---
### :memo: Notes
Make sure to replace `path_to_script_X.py` with the actual path to your Python scripts and `link_to_dataset` with a link to the ZINC250k dataset if you have one.

# :clap: Acknowledgements
Thanks to the creators of the torchdrug library for providing comprehensive tools for drug discovery tasks.
ZINC250k dataset for molecular data.
