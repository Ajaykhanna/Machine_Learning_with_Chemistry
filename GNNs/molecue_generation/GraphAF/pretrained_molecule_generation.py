## Generative Model: GraphAF
# ___
"""
Autoregressive Molecule Generation using GraphAF

This script demonstrates the application of Graph Autoregressive Flow (GraphAF)
for generating molecules from the ZINC250k dataset. The molecules are generated
using a set of predefined hyperparameters without any specific task-driven goal.

The script:
1. Loads the ZINC250k dataset.
2. Defines the GraphAF model.
3. Trains the model.
4. Generates molecules using the trained model.

Usage:
    python <script_name.py>
    
Outputs:
    A `.pkl` model file and generated molecule SMILES strings.
"""

from torchdrug import core, datasets, models, tasks
from torchdrug.layers import distribution
from torch import nn, optim
from collections import defaultdict

dataset = datasets.ZINC250k(
    "~/molecule-datasets/", kekulize=True, atom_feature="symbol"
)

model = models.RGCN(
    input_dim=dataset.num_atom_type,
    num_relation=dataset.num_bond_type,
    hidden_dims=[256, 256, 256],
    batch_norm=True,
)

num_atom_type = dataset.num_atom_type
# add one class for non-edge
num_bond_type = dataset.num_bond_type + 1

node_prior = distribution.IndependentGaussian(
    torch.zeros(num_atom_type), torch.ones(num_atom_type)
)
edge_prior = distribution.IndependentGaussian(
    torch.zeros(num_bond_type), torch.ones(num_bond_type)
)
node_flow = models.GraphAF(model, node_prior, num_layer=12)
edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=12)

task = tasks.AutoregressiveGeneration(
    node_flow, edge_flow, max_node=38, max_edge_unroll=12, criterion="nll"
)

optimizer = optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(
    task, dataset, None, None, optimizer, gpus=(0,), batch_size=128, log_interval=10
)

solver.train(num_epoch=10)
solver.save(
    "/home/akhanna2/data/git_stage/machine_learning_with_chemistry/torch_drug/molecule_generation/graphAF/graphaf_zinc250k_10epoch.pkl"
)

solver.load(
    "/home/akhanna2/data/git_stage/machine_learning_with_chemistry/torch_drug/molecule_generation/graphAF/graphaf_zinc250k_10epoch.pkl"
)
results = task.generate(num_sample=32)
print(results.to_smiles())