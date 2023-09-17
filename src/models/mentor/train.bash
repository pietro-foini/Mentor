#!/bin/bash

# Usage: bash train.bash

# Position.
python train.py --dataset "../../datasets/synthetic/position/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/synthetic/position"

# Topology (tv1).
python train.py --dataset "../../datasets/synthetic/topology/tv1/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/synthetic/topology/tv1"

# Topology (tv2).
python train.py --dataset "../../datasets/synthetic/topology/tv2/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/synthetic/topology/tv2"

# Topology (tv3).
python train.py --dataset "../../datasets/synthetic/topology/tv3/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/synthetic/topology/tv3"

# Centrality (in-degree).
python train.py --dataset "../../datasets/synthetic/centrality/in-degree/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/synthetic/centrality/in-degree"

# Centrality (out-degree).
python train.py --dataset "../../datasets/synthetic/centrality/out-degree/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/synthetic/centrality/out-degree"

# Position and topology.
python train.py --dataset "../../datasets/synthetic/position-topology/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/synthetic/position-topology"

# IMDb.
python train.py --dataset "../../datasets/real-world/IMDb/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/real-world/IMDb"

# Dribbble.
python train.py --dataset "../../datasets/real-world/Dribbble/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/real-world/Dribbble"

# Kaggle.
python train.py --dataset "../../datasets/real-world/Kaggle/data" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "./results/3-channels/real-world/Kaggle"
