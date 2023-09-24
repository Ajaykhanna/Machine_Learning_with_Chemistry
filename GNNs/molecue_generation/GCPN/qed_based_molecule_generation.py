"""
Goal-Directed Molecule Generation using GCPN for the Quantitative Estimation of Drug-likeness (QED) task.

This script demonstrates the application of the Graph Convolutional Policy Network (GCPN) 
for generating molecules with a focus on maximizing their drug-likeness as assessed by the QED score.
The ZINC250k dataset is utilized, and the model is configured for a combination of Proximal Policy Optimization (PPO) 
and negative log-likelihood (NLL) as the criterion for reinforcement learning. 
After model initialization and configuration, a pre-trained model is loaded, further fine-tuned for the QED task, 
and the results are displayed in the SMILES format.

Dependencies:
- torch
- torchdrug

Date: Sep.23.2023 
Place: UC Merced
"""

import torch
from torchdrug import core, datasets, models, tasks
from torch import nn, optim
from collections import defaultdict

# Load the ZINC250k dataset
dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True,
                            atom_feature="symbol")

# Initialize the RGCN model
model = models.RGCN(input_dim=dataset.node_feature_dim,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 256, 256], batch_norm=False)

# Set up the graph generation task using GCPN with a focus on maximizing QED
task = tasks.GCPNGeneration(model, dataset.atom_types,
                            max_edge_unroll=12, max_node=38,
                            task="qed", criterion=("ppo", "nll"),
                            reward_temperature=1,
                            agent_update_interval=3, gamma=0.9)

# Define the Adam optimizer
optimizer = optim.Adam(task.parameters(), lr=1e-5)

# Set up the training engine
solver = core.Engine(task, dataset, None, None, optimizer, 
                    gpus=(0,), batch_size=16, log_interval=10)

# Load a pre-trained model
solver.load("/home/akhanna2/data/git_stage/machine_learning_with_chemistry/torch_drug/molecule_generation/gcpn/graphgeneration/gcpn_zinc250k_1epoch.pkl",
            load_optimizer=False)

# Fine-tune the model with a focus on QED
solver.train(num_epoch=10)

# Save the fine-tuned model
solver.save("/home/akhanna2/data/git_stage/machine_learning_with_chemistry/torch_drug/molecule_generation/gcpn/graphgeneration/gcpn_zinc250k_1epoch_finetune_QED.pkl")

# Generate molecules with the fine-tuned model
results = task.generate(num_sample=32, max_resample=5)

# Display the generated molecules in the SMILES format
print(results.to_smiles())
