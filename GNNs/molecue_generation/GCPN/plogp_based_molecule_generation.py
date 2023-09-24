"""
Goal-Directed Molecule Generation with Reinforcement Learning using GCPN.

This script showcases the application of the Graph Convolutional Policy Network (GCPN)
for goal-directed molecule generation on the ZINC250k dataset. The aim is to use reinforcement
learning to direct the model to generate molecules with specific properties. 
After loading the dataset and initializing the model, we employ the Proximal Policy Optimization (PPO)
algorithm as our RL criterion. The model is then fine-tuned, and the final results 
are displayed in the SMILES format.

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

# Set up the graph generation task using GCPN and configure for PPO-based reinforcement learning
task = tasks.GCPNGeneration(model, dataset.atom_types,
                            max_edge_unroll=12, max_node=38,
                            task="plogp", criterion="ppo",
                            reward_temperature=1,
                            agent_update_interval=3, gamma=0.9)

# Define the Adam optimizer
optimizer = optim.Adam(task.parameters(), lr=1e-5)

# Set up the training engine
solver = core.Engine(task, dataset, None, None, optimizer, 
                    gpus=(0,), batch_size=16, log_interval=10)

# Load a pre-trained model
solver.load("path_to_dump/graphgeneration/gcpn_zinc250k_1epoch.pkl",
            load_optimizer=False)

# Fine-tune the model using reinforcement learning
solver.train(num_epoch=10)

# Save the fine-tuned model
solver.save("path_to_dump/graphgeneration/gcpn_zinc250k_1epoch_finetune.pkl")

# Generate molecules with the fine-tuned model
results = task.generate(num_sample=32, max_resample=5)

# Display the generated molecules in the SMILES format
print(results.to_smiles())