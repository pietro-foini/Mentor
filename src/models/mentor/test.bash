#!/bin/bash

# Usage: bash test.bash

# Position.
for seed in {1..10}; do
    python test.py --dataset "../../datasets/synthetic/position/data" --hyperparameter_path "./results/3-channels/synthetic/position/best_params" --seed "$seed" --workspace "./results/3-channels/synthetic/position"
done

# Topology (tv1).
for seed in {1..10}; do
    python test.py --dataset "../../datasets/synthetic/topology/tv1/data" --hyperparameter_path "./results/3-channels/synthetic/topology/tv1/best_params" --seed "$seed" --workspace "./results/3-channels/synthetic/topology/tv1"
done

# Topology (tv2).
for seed in {1..10}; do
    python test.py --dataset "../../datasets/synthetic/topology/tv2/data" --hyperparameter_path "./results/3-channels/synthetic/topology/tv2/best_params" --seed "$seed" --workspace "./results/3-channels/synthetic/topology/tv2"
done

# Topology (tv3).
for seed in {1..10}; do
    python test.py --dataset "../../datasets/synthetic/topology/tv3/data" --hyperparameter_path "./results/3-channels/synthetic/topology/tv3/best_params" --seed "$seed" --workspace "./results/3-channels/synthetic/topology/tv3"
done

# Centrality (in-degree).
for seed in {1..10}; do
    python test.py --dataset "../../datasets/synthetic/centrality/in-degree/data" --hyperparameter_path "./results/3-channels/synthetic/centrality/in-degree/best_params" --seed "$seed" --workspace "./results/3-channels/synthetic/centrality/in-degree"
done

# Centrality (out-degree).
for seed in {1..10}; do
    python test.py --dataset "../../datasets/synthetic/centrality/out-degree/data" --hyperparameter_path "./results/3-channels/synthetic/centrality/out-degree/best_params" --seed "$seed" --workspace "./results/3-channels/synthetic/centrality/out-degree"
done

# Position and topology.
for seed in {1..10}; do
    python test.py --dataset "../../datasets/synthetic/position-topology/data" --hyperparameter_path "./results/3-channels/synthetic/position-topology/best_params" --seed "$seed" --workspace "./results/3-channels/synthetic/position-topology"
done

# IMDb.
for seed in {1..10}; do
    python test.py --dataset "../../datasets/real-world/IMDb/data" --hyperparameter_path "./results/3-channels/real-world/IMDb/best_params" --seed "$seed" --workspace "./results/3-channels/real-world/IMDb"
done

# Dribbble.
for seed in {1..10}; do
    python test.py --dataset "../../datasets/real-world/Dribbble/data" --hyperparameter_path "./results/3-channels/real-world/Dribbble/best_params" --seed "$seed" --workspace "./results/3-channels/real-world/Dribbble"
done

# Kaggle.
for seed in {1..10}; do
    python test.py --dataset "../../datasets/real-world/Kaggle/data" --hyperparameter_path "./results/3-channels/real-world/Kaggle/best_params" --seed "$seed" --workspace "./results/3-channels/real-world/Kaggle"
done
