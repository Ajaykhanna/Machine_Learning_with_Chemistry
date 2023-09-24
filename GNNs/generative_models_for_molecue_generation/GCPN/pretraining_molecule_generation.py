"""
Graph Generation using GCPN on the ZINC250k Dataset.

This script demonstrates the use of the Graph Convolutional Policy Network (GCPN)
to perform graph generation on the ZINC250k dataset. It initializes an RGCN model,
sets up a graph generation task, trains the model for one epoch, saves the trained model,
and then loads the trained model to generate molecule samples, which are printed in the SMILES format.

Dependencies:
- torch
- torchdrug

Date: Sep.23.2023
Place: UC Merced
"""

import torch
from torch import nn, optim
from torchdrug import core, datasets, models, tasks

# Load the ZINC250k dataset
dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True,
                            atom_feature="symbol")

# Initialize the RGCN (Relational Graph Convolutional Network) model
model = models.RGCN(input_dim=dataset.node_feature_dim,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 256, 256], batch_norm=False)

# Set up the graph generation task using GCPN (Graph Convolutional Policy Network)
task = tasks.GCPNGeneration(model, dataset.atom_types, max_edge_unroll=12,
                            max_node=38, criterion="nll")

# Initialize the Adam optimizer
optimizer = optim.Adam(task.parameters(), lr=1e-3)

# Set up the training engine
solver = core.Engine(task, dataset, None, None, optimizer,
                    gpus=(0,), batch_size=128, log_interval=10)

# Train the model for one epoch
solver.train(num_epoch=1)

# Save the trained model
solver.save("/home/akhanna2/data/git_stage/machine_learning_with_chemistry/torch_drug/molecule_generation/gcpn/graphgeneration/gcpn_zinc250k_1epoch.pkl")

# Load the saved trained model
solver.load("/home/akhanna2/data/git_stage/machine_learning_with_chemistry/torch_drug/molecule_generation/gcpn/graphgeneration/gcpn_zinc250k_1epoch.pkl")

# Generate molecule samples using the trained model
results = task.generate(num_sample=32, max_resample=5)

# Print the generated molecule samples in the SMILES format
print(results.to_smiles())
