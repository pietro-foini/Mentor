import argparse
import logging
import os

import pandas as pd

from utils import load_graph_information, EarlyStoppingCallback
from get_features import HandEngineeredFeatures
from LR.model import train as lr_train
from SVM.model import train as svm_train
from RF.model import train as rf_train
from XGBoost.model import train as xgboost_train
from MLP.model import train as mlp_train

models = {
    "LR": lr_train,
    "SVM": svm_train,
    "RF": rf_train,
    "XGBoost": xgboost_train,
    "MLP": mlp_train,
}


def parse_args():
    """Set argparse arguments for handling training phase on a provided dataset."""
    parser_user = argparse.ArgumentParser(description="Train Logistic Regression (LR) using different random seed.")

    parser_user.add_argument("--dataset_path", type=str, default="../../datasets/synthetic/position/data",
                             help="The path to the folder dataset containing graph's files.")
    parser_user.add_argument("--model_type", type=str, default="LR", help="The model to use.")
    parser_user.add_argument("--test_size", type=float, default=0.2, help="The percentage of samples to use for test.")
    parser_user.add_argument("--early_stop_optuna", type=int, default=80,
                             help="Early stop for Optuna during validation phase.")
    parser_user.add_argument("--k", type=int, default=5, help="The number of folds during the validation phase.")
    parser_user.add_argument("--trials", type=int, default=200,
                             help="The trials of Optuna during the validation phase.")
    parser_user.add_argument("--seeds", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nargs="+",
                             help="Define which seeds to use for reproducibility.")
    parser_user.add_argument("--workspace", type=str, default="results/synthetic/position",
                             help="The name of the folder where the results will be stored.")

    args = parser_user.parse_args()

    return args


def main():
    """Perform training, validation and test phases over a provided dataset using LR."""
    # Get parameters.
    args = parse_args()

    # Create workspace.
    if not os.path.exists(f"./{args.model_type}/{args.workspace}"):
        os.makedirs(f"./{args.model_type}/{args.workspace}")

    # Load dataset information.
    graph, teams_composition, teams_label, nodes_attribute, teams_members, _, _ = load_graph_information(
        args.dataset_path)

    # Get the network features at team level.
    extractor = HandEngineeredFeatures(graph, teams_composition, teams_label)
    # Get X features and y target points.
    inputs, outputs = extractor()

    metrics = {}
    for seed in args.seeds:
        logging.info(f"Experiment seed {seed}:")

        early_stopping = EarlyStoppingCallback(args.early_stop_optuna)
        acc, f1, auroc = models[args.model_type](inputs, outputs, args.test_size, args.k, args.trials, early_stopping,
                                                 nodes_attribute, seed)

        metrics[seed] = {"test_acc": acc, "test_f1": f1, "test_auroc": auroc}

    # Aggregate results.
    results = pd.DataFrame(metrics).T

    with pd.ExcelWriter(f"./{args.model_type}/{args.workspace}/metrics.xlsx") as writer:
        results.to_excel(writer, sheet_name=f"Seeds")
        results.agg(["mean", "std"]).to_excel(writer, sheet_name=f"Overall")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
