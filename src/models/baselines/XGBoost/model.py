from typing import Optional, Callable
import random
import logging
import warnings
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier

# Removes warnings in the current job.
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs.
os.environ["PYTHONWARNINGS"] = "ignore"


def hyper_search(trial, x, y, cv, nodes_attribute, n_classes):
    """Objective function for Optuna optimization."""
    # Search configuration.
    gamma = trial.suggest_uniform("gamma", 0.0, 1.)
    reg_alpha = trial.suggest_uniform("reg_alpha", 0.0, 1.)
    reg_lambda = trial.suggest_uniform("reg_lambda", 0.0, 1.)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_child_weight = trial.suggest_uniform("min_child_weight", 1., 10.)
    learning_rate = trial.suggest_uniform("learning_rate", 0.001, 0.3)
    subsample = trial.suggest_uniform("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.6, 1.0)

    objective = "multi:softmax" if n_classes > 2 else "binary:logistic"

    classifier = XGBClassifier(n_estimators=100,
                               objective=objective,
                               eval_metric="mlogloss",
                               gamma=gamma,
                               reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda,
                               max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               learning_rate=learning_rate,
                               subsample=subsample,
                               colsample_bytree=colsample_bytree,
                               n_jobs=-1)

    # Aggregate nodes attributes based on corresponding team.
    if nodes_attribute is not None:
        agg = trial.suggest_categorical("agg", ["sum", "mean", "min", "max"])
        # Add the node attributes aggregated for team.
        teams_attribute = nodes_attribute.groupby(axis=0, level="team").agg(agg)
        x = pd.merge(x, teams_attribute, left_index=True, right_index=True, how="left")

    # Fit scaler on training data.
    scores = cross_val_score(classifier, x, y, cv=cv, n_jobs=-1)

    return np.mean(scores)


def train(inputs: pd.DataFrame, outputs: pd.DataFrame, test_size: float, k: int, trials_optuna: int,
          callback_optuna: Callable, nodes_attribute: Optional[pd.DataFrame] = None,
          seed: int = 1) -> tuple[float, float, float]:
    """
    Perform training, validation and test phases over a provided dataset using XGBoost.

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

    n_classes = outputs["label"].nunique()

    # Split teams for training and for test.
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_size,
                                                        stratify=outputs, random_state=seed)

    # Ravel label for models.
    y_train, y_test = y_train.values.ravel(), y_test.values.ravel()

    # Optimization.
    logging.info("Hyper-parameter tuning")

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "maximize"
    callback_optuna.direction = direction
    study = optuna.create_study(direction=direction, sampler=TPESampler(multivariate=True, seed=seed))
    study.optimize(lambda x: hyper_search(x, x_train, y_train, cv, nodes_attribute, n_classes),
                   callbacks=[callback_optuna], n_trials=trials_optuna, show_progress_bar=True)

    # Defining parameter range.
    best_params = study.best_params

    logging.info("Test phase")

    if nodes_attribute is not None:
        teams_attribute = nodes_attribute.groupby(axis=0, level="team").agg(best_params["agg"])
        x_train = pd.merge(x_train, teams_attribute, left_index=True, right_index=True, how="left")
        x_test = pd.merge(x_test, teams_attribute, left_index=True, right_index=True, how="left")
        del best_params["agg"]

    objective = "multi:softmax" if n_classes > 2 else "binary:logistic"

    model = XGBClassifier(n_estimators=100, objective=objective, eval_metric="mlogloss", **best_params)
    model.fit(x_train, y_train)

    # Test.
    accuracy_test = accuracy_score(y_test, model.predict(x_test))
    f1_test = f1_score(y_test, model.predict(x_test), average="micro")
    auroc_test = roc_auc_score(y_test, model.predict_proba(x_test), average="macro", multi_class="ovr")

    return accuracy_test, f1_test, auroc_test
