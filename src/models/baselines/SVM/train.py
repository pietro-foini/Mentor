import argparse
import random
import logging
import warnings
import os

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler

from src.models.load_dataset import load_graph_information
from src.models.baselines.utils import HandEngineeredFeatures, EarlyStoppingCallback

# Removes warnings in the current job.
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs.
os.environ["PYTHONWARNINGS"] = "ignore"

scaler_dict = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
    "QuantileTransformer": QuantileTransformer(),
    "RobustScaler": RobustScaler()
}


def parse_args():
    """
    This function set argparse arguments for performing training, validation and test phases over a provided dataset.

    :return: argparse arguments
    """
    parser_user = argparse.ArgumentParser(description="Train Support Vector Machine (SVM) using different random seed.")

    parser_user.add_argument("--dataset_path", type=str, default="../../../datasets/real-world/IMDb/data",
                             help="The path to the folder dataset containing graph's files.")
    parser_user.add_argument("--test_size", type=float, default=0.2, help="The percentage of samples to use for test.")
    parser_user.add_argument("--early_stop_optuna", type=int, default=80,
                             help="Early stop for Optuna during validation phase.")
    parser_user.add_argument("--k", type=int, default=5, help="The number of folds during the validation phase.")
    parser_user.add_argument("--trials", type=int, default=200,
                             help="The trials of Optuna during the validation phase.")
    parser_user.add_argument("--seeds", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nargs="+",
                             help="Define which seeds to use for reproducibility.")
    parser_user.add_argument("--workspace", type=str, default="results/IMDb",
                             help="The name of the folder where the results will be stored.")

    args = parser_user.parse_args()

    return args


def hyper_search(trial, x, y, cv, nodes_attribute):
    # Search configuration.
    c = trial.suggest_categorical("C", [0.1, 1, 10, 100, 1000])
    gamma = trial.suggest_categorical("gamma", [1, 0.1, 0.01, 0.001, 0.0001])
    norm_func = trial.suggest_categorical("norm_func", ["MinMaxScaler", "StandardScaler",
                                                        "QuantileTransformer", "RobustScaler"])

    classifier = SVC(decision_function_shape="ovr", class_weight="balanced", gamma=gamma,
                     C=c, kernel="rbf")

    # Aggregate nodes attributes based on corresponding team.
    if nodes_attribute is not None:
        agg = trial.suggest_categorical("agg", ["sum", "mean", "min", "max"])
        # Add the node attributes aggregated for team.
        teams_attribute = nodes_attribute.groupby(axis=0, level="team").agg(agg)
        x = pd.merge(x, teams_attribute, left_index=True, right_index=True, how="left")

    # Fit scaler on training data.
    pipeline = Pipeline([("transformer", scaler_dict[norm_func]), ("estimator", classifier)])
    scores = cross_val_score(pipeline, x, y, cv=cv, n_jobs=-1)

    return np.mean(scores)


def main():
    """
    Perform training, validation and test phases over a provided dataset using Logistic Regression.

    :return:
    """
    # Get parameters.
    args = parse_args()

    # Create workspace.
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)

    # Load dataset information.
    graph, teams_composition, teams_label, nodes_attribute = load_graph_information(args.dataset_path)

    # Get the network features at team level.
    extractor = HandEngineeredFeatures(graph, teams_composition, teams_label)
    # Get X features and y target points.
    inputs, outputs = extractor()

    metrics, confusion_matrices = {}, {}
    for seed in args.seeds:
        logging.info(f"Experiment seed {seed}:")

        # Fix random seed.
        random.seed(seed)
        np.random.seed(seed)

        # Split teams for training and for test.
        x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=args.test_size,
                                                            stratify=outputs, random_state=seed)

        # Ravel label for models.
        y_train, y_test = y_train.values.ravel(), y_test.values.ravel()

        # Optimization.
        logging.info("Hyper-parameter tuning")

        cv = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=seed)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        direction = "maximize"
        study = optuna.create_study(direction=direction, sampler=TPESampler(multivariate=True, seed=seed))
        early_stopping = EarlyStoppingCallback(args.early_stop_optuna, direction=direction)
        study.optimize(lambda x: hyper_search(x, x_train, y_train, cv, nodes_attribute),
                       callbacks=[early_stopping], n_trials=args.trials, show_progress_bar=True)

        # Defining parameter range.
        best_params = study.best_params

        logging.info("Test phase")

        if nodes_attribute is not None:
            teams_attribute = nodes_attribute.groupby(axis=0, level="team").agg(best_params["agg"])
            x_train = pd.merge(x_train, teams_attribute, left_index=True, right_index=True, how="left")
            x_test = pd.merge(x_test, teams_attribute, left_index=True, right_index=True, how="left")
            del best_params["agg"]

        # Fit scaler on training data.
        norm = scaler_dict[best_params["norm_func"]].fit(x_train)
        # Transform training data.
        x_train = norm.transform(x_train)
        # Transform testing data.
        x_test = norm.transform(x_test)

        del best_params["norm_func"]

        model = SVC(decision_function_shape="ovr", class_weight="balanced", kernel="rbf", probability=True,
                    **best_params)
        model.fit(x_train, y_train)

        # Test.
        accuracy_test = accuracy_score(y_test, model.predict(x_test))
        f1_test = f1_score(y_test, model.predict(x_test), average="micro")
        auroc_test = roc_auc_score(y_test, model.predict_proba(x_test), average="macro", multi_class="ovr")
        conf_test = confusion_matrix(y_test, model.predict(x_test))

        metrics[seed] = {"Accuracy": accuracy_test, "F1": f1_test, "AUROC": auroc_test}
        confusion_matrices[seed] = conf_test

    # Aggregate results.
    results = pd.DataFrame(metrics).T

    # Save results.
    with pd.ExcelWriter(f"{args.workspace}/confusion_matrix.xlsx") as writer:
        for seed, conf_mtrx in confusion_matrices.items():
            pd.DataFrame(conf_mtrx).to_excel(writer, sheet_name=f"Seed {seed}")

    with pd.ExcelWriter(f"{args.workspace}/metrics.xlsx") as writer:
        results.to_excel(writer, sheet_name=f"Seeds")
        results.agg(["mean", "std"]).to_excel(writer, sheet_name=f"Overall")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
