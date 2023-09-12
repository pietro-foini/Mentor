import argparse
import random
import logging
import warnings
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torchmetrics import Accuracy, F1Score, AUROC
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch_geometric.data import Data, DataLoader

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
    parser_user = argparse.ArgumentParser(description="Train Multi Layer Perceptron (MLP) using different random seed.")

    parser_user.add_argument("--dataset_path", type=str, default="../../../datasets/real-world/IMDb/data",
                             help="The path to the folder dataset containing graph's files.")
    parser_user.add_argument("--test_size", type=float, default=0.2, help="The percentage of samples to use for test.")
    parser_user.add_argument("--early_stop_optuna", type=int, default=80,
                             help="Early stop for Optuna during validation phase.")
    parser_user.add_argument("--k", type=int, default=5, help="The number of folds during the validation phase.")
    parser_user.add_argument("--trials", type=int, default=2,
                             help="The trials of Optuna during the validation phase.")
    parser_user.add_argument("--seeds", type=int, default=[1], nargs="+",
                             help="Define which seeds to use for reproducibility.")
    parser_user.add_argument("--workspace", type=str, default="results/IMDb",
                             help="The name of the folder where the results will be stored.")

    args = parser_user.parse_args()

    return args


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            # Create the MLP by stacking components.
            layers.append(Linear(input_dim, hidden_dim))
            layers.append(BatchNorm1d(hidden_dim))
            layers.append(ReLU(inplace=True))
            layers.append(Dropout(p=dropout))

            input_dim = hidden_dim

        # Last linear layer to map the hidden space to desired output dimension.
        layers.append(Linear(input_dim, output_dim))

        # Create pytorch nn module by stacking the previous modules.
        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LightningNet(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=2, dropout=0.5, lr=0.01, **kwargs):
        super().__init__()

        self.lr = lr

        # Initialize the MLP.
        self.model = Net(input_dim, hidden_dim, output_dim, num_layers, dropout)

        # Evaluation metrics.
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.test_f1 = F1Score(num_classes=output_dim)
        self.test_auroc = AUROC(num_classes=output_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x)
        train_loss = F.cross_entropy(y_hat, batch.y)
        train_acc = self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log("train_acc", train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x)
        val_loss = F.cross_entropy(y_hat, batch.y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch.x)
        test_acc = self.test_acc(y_hat.softmax(dim=-1), batch.y)
        test_f1 = self.test_f1(y_hat.softmax(dim=-1).argmax(-1), batch.y)
        test_auroc = self.test_auroc(y_hat.softmax(dim=-1), batch.y)

        self.log("test_acc", test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_f1", test_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_auroc", test_auroc, prog_bar=True, on_step=False, on_epoch=True)
        return test_acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def create_dataloader(x, y, batch_size=64, shuffle=True):
    return DataLoader([Data(x=torch.FloatTensor(xx).view(1,-1), y=torch.tensor(yy)) for xx,yy in zip(x,y)],
                      batch_size=batch_size, shuffle=shuffle)


def hyper_search(trial, x, y, k, nodes_attribute, n_classes, seed):
    # Fix seed for reproducibility.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Search configuration.
    epochs = int(trial.suggest_discrete_uniform("epochs", 20, 100, 2))
    dropout = trial.suggest_discrete_uniform("dropout", 0.2, 0.8, 0.05)
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    norm_func = trial.suggest_categorical("norm_func", ["MinMaxScaler", "StandardScaler",
                                                        "QuantileTransformer", "RobustScaler"])

    # Aggregate nodes attributes based on corresponding team.
    if nodes_attribute is not None:
        agg = trial.suggest_categorical("agg", ["sum", "mean", "min", "max"])
        # Add the node attributes aggregated for team.
        teams_attribute = nodes_attribute.groupby(axis=0, level="team").agg(agg)
        x = pd.merge(x, teams_attribute, left_index=True, right_index=True, how="left")

    # Train and validation: StratifiedKFold.
    sss = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    train_teams = x.index.tolist()
    train_labels = y.values.ravel().tolist()

    losses = []
    for train_idx, val_idx in sss.split(train_teams, train_labels):
        train_idx, val_idx = np.array(train_teams)[train_idx].tolist(), np.array(train_teams)[val_idx].tolist()
        x_train, y_train = x.loc[train_idx].values, y.loc[train_idx].values
        x_val, y_val = x.loc[val_idx].values, y.loc[val_idx].values

        # Normalization.
        norm = globals()[norm_func]().fit(x_train)
        x_train = norm.transform(x_train)
        x_val = norm.transform(x_val)

        # Create DataLoaders.
        train_loader = create_dataloader(x_train, y_train, batch_size=64)
        val_loader = create_dataloader(x_val, y_val, batch_size=x_val.shape[0])

        # Training and validation.
        model = LightningNet(x_train.shape[1], n_classes, hidden_dim=hidden_dim, num_layers=num_layers,
                             dropout=dropout, lr=lr)
        trainer = Trainer(gpus=1, max_epochs=epochs, checkpoint_callback=False,
                          logger=False, weights_summary=None, progress_bar_refresh_rate=0)
        trainer.fit(model, train_loader)
        out = trainer.validate(model, val_dataloaders=val_loader, verbose=False)
        losses.append(out[0]["val_loss"])

    return np.mean(losses)


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

    n_classes = outputs["label"].nunique()

    metrics = {}
    for seed in args.seeds:
        logging.info(f"Experiment seed {seed}:")

        # Fix random seed.
        random.seed(seed)
        np.random.seed(seed)

        # Split teams for training and for test.
        x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=args.test_size,
                                                            stratify=outputs, random_state=seed)

        # Optimization.
        logging.info("Hyper-parameter tuning")

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        direction = "minimize"
        study = optuna.create_study(direction=direction, sampler=TPESampler(multivariate=True, seed=seed))
        early_stopping = EarlyStoppingCallback(args.early_stop_optuna, direction=direction)
        study.optimize(lambda x: hyper_search(x, x_train, y_train, args.k, nodes_attribute, n_classes, seed),
                       n_trials=args.trials, callbacks=[early_stopping], show_progress_bar=True)

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

        # Create DataLoaders.
        train_loader = create_dataloader(x_train, y_train.values, batch_size=64)
        test_loader = create_dataloader(x_test, y_test.values, batch_size=x_test.shape[0])

        # Holdout test.
        test_acc, test_f1, test_auroc = [], [], []
        for i in range(10):
            # Fix seed for reproducibility.
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = LightningNet(x_train.shape[1], n_classes, **best_params)

            trainer = Trainer(gpus=1, max_epochs=int(best_params["epochs"]), checkpoint_callback=False,
                              logger=False, weights_summary=None, progress_bar_refresh_rate=0)
            trainer.fit(model, train_loader)
            result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
            test_acc.append(result[0]["test_acc"])
            test_f1.append(result[0]["test_f1"])
            test_auroc.append(result[0]["test_auroc"])

        metrics[seed] = {"Accuracy": np.mean(test_acc), "F1": np.mean(test_f1), "AUROC": np.mean(test_auroc)}

    # Aggregate results.
    results = pd.DataFrame(metrics).T

    # Save results.
    with pd.ExcelWriter(f"{args.workspace}/metrics.xlsx") as writer:
        results.to_excel(writer, sheet_name=f"Seeds")
        results.agg(["mean", "std"]).to_excel(writer, sheet_name=f"Overall")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
