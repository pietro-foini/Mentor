import os
import pickle
import logging
import random
import argparse
import warnings

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import torch
from pytorch_lightning import Trainer
from torch_geometric.data import DataLoader
import optuna
from optuna.samplers import TPESampler

from model.model import SingleFramework
from model.hetero_data import DataObject
from utils import load_graph_information, EarlyStoppingCallback

torch.set_default_dtype(torch.float64)

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
    """Set argparse arguments for handling training phase on a provided dataset."""
    parser_user = argparse.ArgumentParser(description="Train Mentor using different random seed.")

    parser_user.add_argument("--dataset_path", type=str, default="../../datasets/synthetic/position/data",
                             help="The path to the folder dataset containing graph's files.")
    parser_user.add_argument("--test_size", type=float, default=0.2, help="The percentage of samples to use for test.")
    parser_user.add_argument("--early_stop_optuna", type=int, default=80,
                             help="Early stop for Optuna during validation phase.")
    parser_user.add_argument("--k", type=int, default=5, help="The number of folds during the validation phase.")
    parser_user.add_argument("--trials", type=int, default=200,
                             help="The trials of Optuna during the validation phase.")
    parser_user.add_argument("--seeds", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nargs="+",
                             help="Define which seeds to use for reproducibility.")
    parser_user.add_argument("--workspace", type=str, default="results/position",
                             help="The name of the folder where the results will be stored.")

    args = parser_user.parse_args()

    return args


def hyper_search(trial, data, k, seed):
    """Objective function for Optuna optimization."""
    # Fix seed for reproducibility.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Flow convolution (topology and centrality channels).
    flow_conv_t = trial.suggest_categorical("flow_conv_t", ["source_to_target", "target_to_source"])
    flow_conv_c = trial.suggest_categorical("flow_conv_c", ["source_to_target", "target_to_source"])
    # Aggregation convolution (topology, centrality and position channels).
    agg_conv_t = trial.suggest_categorical("agg_conv_t", ["sum", "mean", "max", "min"])
    agg_conv_c = trial.suggest_categorical("agg_conv_c", ["sum", "mean", "max", "min"])
    agg_conv_p = trial.suggest_categorical("agg_conv_p", ["sum", "mean", "max", "min"])
    # Aggregation team (topology channel).
    agg_team_t = trial.suggest_categorical("agg_team_t", ["sum", "mean", "max", "min"])
    # Hidden channels.
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    # Dropout.
    dropout_t = trial.suggest_discrete_uniform("dropout_t", 0.2, 0.8, 0.05)
    dropout_c = trial.suggest_discrete_uniform("dropout_c", 0.2, 0.8, 0.05)
    dropout_a = trial.suggest_discrete_uniform("dropout_a", 0.2, 0.8, 0.05)
    # Epochs.
    epochs = trial.suggest_discrete_uniform("epochs", 20, 100, 2)
    # Learning parameters.
    lr_base = trial.suggest_loguniform("lr_base", 1e-5, 1e-1)
    lr_swa = trial.suggest_loguniform("lr_swa", 1e-5, 1e-1)
    swa_start = trial.suggest_discrete_uniform("swa_start", 0.6, 0.95, 0.05)
    swa_freq = trial.suggest_int("swa_freq", 1, 20)

    # Train and validation: StratifiedKFold.
    sss = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    losses = []
    for train_idx, val_idx in sss.split(data.train_teams, data.train_labels):
        # Split teams for training and validation.
        mask_train = np.full(data.n_teams, False)
        mask_val = np.full(data.n_teams, False)

        mask_train[np.array(data.train_teams)[train_idx]] = True
        mask_val[np.array(data.train_teams)[val_idx]] = True

        data.mask_train = torch.tensor(mask_train)
        data.mask_val = torch.tensor(mask_val)

        # Normalization attributes.
        if data["topology"].norm:
            # Normalization.
            norm_func = trial.suggest_categorical("norm_func", ["MinMaxScaler", "StandardScaler",
                                                                "RobustScaler", "QuantileTransformer"])
            # Fit scaler on training data.
            x_tmp = scaler_dict[norm_func].fit_transform(data["topology"].x)
            data["topology"].x_norm = torch.tensor(x_tmp)
        else:
            x_tmp = np.copy(data["topology"].x)
            data["topology"].x_norm = torch.tensor(x_tmp)

        # Training and validation.
        model = SingleFramework(input_dim_t=data["topology"].x.shape[1], input_dim_c=data["centrality"].x.shape[1],
                                input_dim_p=data["position"].x.shape[1], n_anchorsets=data["position"].n_anchorsets,
                                out_dim=data.n_classes, hidden_dim=hidden_dim, flow_conv_T=flow_conv_t,
                                flow_conv_c=flow_conv_c, agg_conv_t=agg_conv_t, agg_conv_C=agg_conv_c,
                                agg_conv_p=agg_conv_p, agg_team_t=agg_team_t, dropout_t=dropout_t, dropout_c=dropout_c,
                                dropout_a=dropout_a, epochs=epochs, lr_base=lr_base, lr_swa=lr_swa, swa_start=swa_start,
                                swa_freq=swa_freq)

        train_loader = DataLoader([data], batch_size=1, shuffle=False)
        trainer = Trainer(gpus=1, max_epochs=int(epochs), checkpoint_callback=False,
                          logger=False, weights_summary=None, progress_bar_refresh_rate=0)
        trainer.fit(model, train_loader)
        out = trainer.validate(model, val_dataloaders=train_loader, verbose=False)
        losses.append(out[0]["val_loss"])

    return np.mean(losses)


def main():
    """Train and validate over a provided dataset using Mentor saving the best parameters for each seed."""
    # Get parameters.
    args = parse_args()

    # Create workspace.
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)
        os.mkdir(f"{args.workspace}/best_params")

    # Load dataset information.
    graph, teams_composition, teams_label, nodes_attribute, teams_members, _, _ = load_graph_information(
        args.dataset_path)

    # Define the teams and the team labels.
    teams = list(teams_label.keys())
    y = list(teams_label.values())
    # Define the number of teams and the number of unique labels/classes.
    n_teams = len(teams)
    # Get nodes that not belong to the teams: None value in correspondence of no team belonging.
    nodes_not_belong_to_teams = [node for node, teams in teams_composition.items() if teams is None]

    random.seed(0)
    np.random.seed(0)

    # Initialization of the HeteroData torch object.
    data = DataObject(graph, teams_composition, teams_members, teams_label, nodes_not_belong_to_teams, nodes_attribute)
    # Perform preprocessing of the 3-channels.
    data = data(topology=True, centrality=True, position=True)

    for seed in args.seeds:
        logging.info(f"Experiment seed {seed}:")

        # Split teams for training and for test.
        teams_mask_train = np.full(n_teams, False)
        teams_mask_test = np.full(n_teams, False)

        train_teams, test_teams, train_labels, test_labels = train_test_split(teams, y, test_size=args.test_size,
                                                                              stratify=y, random_state=seed)

        teams_mask_train[train_teams] = True
        teams_mask_test[test_teams] = True

        # Set training teams into HeteroData.
        data.train_teams, data.train_labels = train_teams, train_labels

        # Optimization.
        logging.info("Hyper-parameter tuning")

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        direction = "minimize"
        study = optuna.create_study(direction=direction, sampler=TPESampler(multivariate=True, seed=seed))
        early_stopping = EarlyStoppingCallback(args.early_stop_optuna, direction=direction)
        study.optimize(lambda x: hyper_search(x, data, args.k, seed), n_trials=args.trials, callbacks=[early_stopping],
                       show_progress_bar=True)

        # Defining best parameter range.
        best_params = study.best_params

        with open(f"{args.workspace}/best_params/{seed}.pkl", "wb") as fp:
            pickle.dump(best_params, fp)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
