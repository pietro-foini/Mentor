#!/bin/bash

# Usage: bash main.bash

# Position.
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/position/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/position"
done

# Centrality (in-degree).
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/centrality/in-degree/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/centrality/in-degree"
done

# Centrality (out-degree).
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/centrality/out-degree/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/centrality/out-degree"
done

# Topology (tv1).
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/topology/tv1/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/topology/tv1"
done

# Topology (tv2).
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/topology/tv2/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/topology/tv2"
done

# Topology (tv3).
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/topology/tv3/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/topology/tv3"
done

# Position and topology.
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/position-topology/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/position-topology"
done

# IMDb.
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/real-world/IMDb/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/real-world/IMDb"
done

# Dribbble.
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/real-world/Dribbble/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/real-world/Dribbble"
done

# Kaggle.
for model in "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/real-world/Kaggle/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/real-world/Kaggle"
done
