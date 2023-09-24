## Goal-Directed Molecule Generation with Reinforcement Learning: GraphAF --> pLOGP
# ___

import torch
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
    node_flow,
    edge_flow,
    max_node=38,
    max_edge_unroll=12,
    task="plogp",
    criterion="ppo",
    reward_temperature=20,
    baseline_momentum=0.9,
    agent_update_interval=5,
    gamma=0.9,
)


optimizer = optim.Adam(task.parameters(), lr=1e-5)
solver = core.Engine(
    task, dataset, None, None, optimizer, gpus=(0,), batch_size=64, log_interval=10
)

solver.load(
    "/home/akhanna2/data/git_stage/machine_learning_with_chemistry/torch_drug/molecule_generation/graphAF/graphaf_zinc250k_10epoch.pkl",
    load_optimizer=False,
)

# RL finetuning
solver.train(num_epoch=10)
solver.save(
    "/home/akhanna2/data/git_stage/machine_learning_with_chemistry/torch_drug/molecule_generation/graphAF/graphaf_zinc250k_10epoch_finetune_pLOGP.pkl"
)

solver.load("/home/akhanna2/data/git_stage/machine_learning_with_chemistry/torch_drug/molecule_generation/graphAF/graphaf_zinc250k_10epoch_finetune_pLOGP.pkl")
results = task.generate(num_sample=32)
print(results.to_smiles())


