#!/bin/bash

# Usage: bash main.bash

EXTRA_FEATURES=false

# Parse command-line arguments for --extra_features
while [ "$#" -gt 0 ]; do
  case "$1" in
    --extra_features)
      EXTRA_FEATURES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Position.
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/position/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/position" --extra_features $EXTRA_FEATURES
done

# Centrality (in-degree).
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/centrality/in-degree/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/centrality/in-degree" --extra_features $EXTRA_FEATURES
done

# Centrality (out-degree).
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/centrality/out-degree/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/centrality/out-degree" --extra_features $EXTRA_FEATURES
done

# Topology (tv1).
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/topology/tv1/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/topology/tv1" --extra_features $EXTRA_FEATURES
done

# Topology (tv2).
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/topology/tv2/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/topology/tv2" --extra_features $EXTRA_FEATURES
done

# Topology (tv3).
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/topology/tv3/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/topology/tv3" --extra_features $EXTRA_FEATURES
done

# Position and topology.
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/synthetic/position-topology/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/synthetic/position-topology" --extra_features $EXTRA_FEATURES
done

# IMDb.
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/real-world/IMDb/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/real-world/IMDb" --extra_features $EXTRA_FEATURES
done

# Dribbble.
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/real-world/Dribbble/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/real-world/Dribbble" --extra_features $EXTRA_FEATURES
done

# Kaggle.
for model in "MLP" "LR" "SVM" "RF" "XGBoost"; do
    python main.py --dataset "../../datasets/real-world/Kaggle/data" --model_type "$model" --test_size 0.2 --early_stop_optuna 80 --k 5 --trials 200 --workspace "results/real-world/Kaggle" --extra_features $EXTRA_FEATURES
done
