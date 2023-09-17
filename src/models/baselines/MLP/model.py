from typing import Optional, Callable
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
    """Objective function for Optuna optimization."""
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
        norm = scaler_dict[norm_func].fit(x_train)
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


def train(inputs: pd.DataFrame, outputs: pd.DataFrame, test_size: float, k: int, trials_optuna: int,
          callback_optuna: Callable, nodes_attribute: Optional[pd.DataFrame] = None,
          seed: int = 1) -> tuple[float, float, float]:
    """
    Perform training, validation and test phases over a provided dataset using MLP.

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

    # Optimization.
    logging.info("Hyper-parameter tuning")

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "minimize"
    callback_optuna.direction = direction
    study = optuna.create_study(direction=direction, sampler=TPESampler(multivariate=True, seed=seed))
    study.optimize(lambda x: hyper_search(x, x_train, y_train, k, nodes_attribute, n_classes, seed),
                   n_trials=trials_optuna, callbacks=[callback_optuna], show_progress_bar=True)

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

    # Test.
    accuracy_test = np.mean(test_acc)
    f1_test = np.mean(test_f1)
    auroc_test = np.mean(test_auroc)

    return accuracy_test, f1_test, auroc_test
