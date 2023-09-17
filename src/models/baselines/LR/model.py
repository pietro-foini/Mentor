import logging
import os
import random
import warnings
from typing import Callable, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler

# Removes warnings in the current job.
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs.
os.environ["PYTHONWARNINGS"] = "ignore"

scaler_dict = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
    "QuantileTransformer": QuantileTransformer(),
    "RobustScaler": RobustScaler(),
}


def hyper_search(trial, x, y, cv, nodes_attribute):
    """Objective function for Optuna optimization."""
    # Search configuration.
    c = trial.suggest_categorical("C", [0.1, 1, 10, 100, 1000])
    norm_func = trial.suggest_categorical(
        "norm_func", ["MinMaxScaler", "StandardScaler", "QuantileTransformer", "RobustScaler"]
    )

    classifier = LogisticRegression(C=c, class_weight="balanced")

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


def train(
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
    test_size: float,
    k: int,
    trials_optuna: int,
    callback_optuna: Callable,
    nodes_attribute: Optional[pd.DataFrame] = None,
    seed: int = 1,
) -> tuple[float, float, float]:
    """
    Perform training, validation and test phases over a provided dataset using LR.

    :param inputs: the hand engineered features extracted by the graph for each team
    :param outputs: the labels of the teams
    :param test_size: the percentage size of the test set
    :param k: the number of k-fold validation
    :param trials_optuna: the numbers of trials for optuna optimization
    :param callback_optuna: the early stopping callback for optuna optimization
    :param nodes_attribute: a dataframe containing the optional node features
    :param seed: the seed for reproducibility
    :return: the accuracy, f1 score and auroc on test set
    """
    # Fix random seed.
    random.seed(seed)
    np.random.seed(seed)

    # Split teams for training and for test.
    x_train, x_test, y_train, y_test = train_test_split(
        inputs, outputs, test_size=test_size, stratify=outputs, random_state=seed
    )

    # Ravel label for models.
    y_train, y_test = y_train.values.ravel(), y_test.values.ravel()

    # Optimization.
    logging.info("Hyper-parameter tuning")

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "maximize"
    callback_optuna.direction = direction
    study = optuna.create_study(direction=direction, sampler=TPESampler(multivariate=True, seed=seed))
    study.optimize(
        lambda x: hyper_search(x, x_train, y_train, cv, nodes_attribute),
        callbacks=[callback_optuna],
        n_trials=trials_optuna,
        show_progress_bar=True,
    )

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

    model = LogisticRegression(**best_params, class_weight="balanced")
    model.fit(x_train, y_train)

    # Test.
    accuracy_test = accuracy_score(y_test, model.predict(x_test))
    f1_test = f1_score(y_test, model.predict(x_test), average="micro")
    auroc_test = roc_auc_score(y_test, model.predict_proba(x_test), average="macro", multi_class="ovr")

    return accuracy_test, f1_test, auroc_test
