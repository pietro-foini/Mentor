import argparse
import logging
import os
import pickle
import random
import shutil
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from model.hetero_data import DataObject
from model.model import SingleFramework
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from torch.utils import tensorboard
from torch_geometric.data import DataLoader
from utils import load_graph_information

torch.set_default_dtype(torch.float64)

matplotlib.style.use("seaborn")

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


def parse_args():
    """Set argparse arguments for handling test phase."""
    parser_user = argparse.ArgumentParser(description="Train Mentor using different random seed.")

    parser_user.add_argument(
        "--dataset_path",
        type=str,
        default="../../datasets/synthetic/position/data",
        help="The path to the folder dataset containing graph's files.",
    )
    parser_user.add_argument(
        "--hyperparameter_path",
        type=str,
        default="./results/position/best_params",
        help="The path to the folder containing best parameters for the provided seed.",
    )
    parser_user.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Define which seeds to use for reproducibility.",
    )
    parser_user.add_argument(
        "--workspace",
        type=str,
        default="results/position",
        help="The name of the folder where the results will be stored.",
    )

    args = parser_user.parse_args()

    return args


def create_workspace(path: str):
    """Create workspace for storing output results."""
    if not os.path.exists(f"{path}/model_weights"):
        os.mkdir(f"{path}/model_weights")

    if not os.path.exists(f"{path}/data_objects"):
        os.mkdir(f"{path}/data_objects")

    if not os.path.exists(f"{path}/tensorboard"):
        os.mkdir(f"{path}/tensorboard")

    if not os.path.exists(f"{path}/results_test"):
        os.mkdir(f"{path}/results_test")

    if not os.path.exists(f"{path}/confusion_matrices_test"):
        os.mkdir(f"{path}/confusion_matrices_test")

    if not os.path.exists(f"{path}/confusion_matrices_test/heatmaps"):
        os.mkdir(f"{path}/confusion_matrices_test/heatmaps")

    if not os.path.exists(f"{path}/channels_attention_coefficients"):
        os.mkdir(f"{path}/channels_attention_coefficients")

    if not os.path.exists(f"{path}/channels_attention_coefficients/heatmaps"):
        os.mkdir(f"{path}/channels_attention_coefficients/heatmaps")

    if not os.path.exists(f"{path}/channels_attention_coefficients/barplots"):
        os.mkdir(f"{path}/channels_attention_coefficients/barplots")

    if not os.path.exists(f"{path}/topology_attention_coefficients"):
        os.mkdir(f"{path}/topology_attention_coefficients")


def main():
    """Train and validate over a provided dataset using Mentor saving the best parameters for each seed."""
    # Get parameters.
    args = parse_args()

    # Create workspace.
    create_workspace(args.workspace)

    # Load parameters obtained from selected previous hyperparameter tuning.
    with open(f"{args.hyperparameter_path}/{args.seed}.pkl", "rb") as f:
        best_params = pickle.load(f)

        # Load dataset information.
    (
        graph,
        teams_composition,
        teams_label,
        nodes_attribute,
        teams_members,
        _,
        _,
    ) = load_graph_information(args.dataset_path)

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
    HeteroData = DataObject(
        graph,
        teams_composition,
        teams_members,
        teams_label,
        nodes_not_belong_to_teams,
        nodes_attribute,
    )
    # Perform preprocessing of the 3-channels.
    data = HeteroData(topology=True, centrality=True, position=True)

    # Split teams for training and for test.
    teams_mask_train = np.full(n_teams, False)
    teams_mask_test = np.full(n_teams, False)

    train_teams, test_teams, train_labels, test_labels = train_test_split(teams, y, test_size=0.2, stratify=y, random_state=args.seed)

    teams_mask_train[train_teams] = True
    teams_mask_test[test_teams] = True

    data.train_teams, data.train_labels = train_teams, train_labels

    logging.info(f"Test phase for seed {args.seed}:")

    # Fix seed for reproducibility.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Tensorboard.
    if not os.path.exists(f"{args.workspace}/tensorboard/{args.seed}"):
        writer = tensorboard.SummaryWriter(f"{args.workspace}/tensorboard/{args.seed}")
    else:
        shutil.rmtree(f"{args.workspace}/tensorboard/{args.seed}")
        writer = tensorboard.SummaryWriter(f"{args.workspace}/tensorboard/{args.seed}")

    # Train and test mask.
    data.mask_train = torch.tensor(teams_mask_train)
    data.mask_test = torch.tensor(teams_mask_test)

    # Normalization attributes.
    if data["topology"].norm:
        # Fit scaler on training data.
        x_tmp = scaler_dict[best_params["norm_func"]].fit_transform(data["topology"].x)
        data["topology"].x_norm = torch.tensor(x_tmp)
    else:
        x_tmp = np.copy(data["topology"].x)
        data["topology"].x_norm = torch.tensor(x_tmp)

    # Training and test.
    model = SingleFramework(
        input_dim_t=data["topology"].x.shape[1],
        input_dim_c=data["centrality"].x.shape[1],
        input_dim_p=data["position"].x.shape[1],
        n_anchorsets=data["position"].n_anchorsets,
        out_dim=data.n_classes,
        tensorboard=writer,
        **best_params,
    )

    train_loader = DataLoader([data], batch_size=1, shuffle=False)
    trainer = Trainer(
        gpus=1,
        max_epochs=int(best_params["epochs"]),
        checkpoint_callback=False,
        logger=False,
        weights_summary=None,
    )
    trainer.fit(model, train_loader)
    result = trainer.test(model, test_dataloaders=train_loader)[0]
    test_accuracy = round(result["test_acc"] * 100, 2)

    writer.close()

    # Store metric results of the current seed.
    results_test = pd.DataFrame(result, index=[args.seed])

    logging.info(f"Accuracy: {test_accuracy}")

    # Save results on test set.
    results_test.to_csv(f"{args.workspace}/results_test/{args.seed}.csv")
    # Save model weights.
    trainer.save_checkpoint(f"{args.workspace}/model_weights/{args.seed}.ckpt")
    # Save Data object.
    torch.save(data, f"{args.workspace}/data_objects/{args.seed}")

    # Get attention coefficients at team level for each channel.
    att_prob = model.attentions_channels.detach().cpu().numpy()
    df_att_channels = pd.DataFrame(att_prob, columns=model.channels)
    df_att_channels["label"] = pd.Series(teams_label)

    df_att_channels.to_csv(f"{args.workspace}/channels_attention_coefficients/{args.seed}.csv")

    # Get attention coefficients at team level for each channel (heatmaps).
    fig, axs = plt.subplots(figsize=(3, 5), ncols=1, nrows=1)
    sns.heatmap(
        df_att_channels[model.channels],
        vmin=0,
        vmax=1,
        cmap="seismic",
        cbar_kws={"label": "Attention coefficients, $\gamma^{(i)}$"},
        ax=axs,
    )
    axs.set_ylabel("TeamID")
    axs.set_xlabel("Channels")

    fig.savefig(
        f"{args.workspace}/channels_attention_coefficients/heatmaps/{args.seed}_acc_{test_accuracy}.png",
        bbox_inches="tight",
        dpi=300,
    )

    # Get attention coefficients for each channel in relation to the label (barplot).
    fig, axs = plt.subplots(figsize=(5, 3), ncols=1, nrows=1)

    m = pd.melt(
        df_att_channels,
        id_vars=["label"],
        value_vars=["topology", "centrality", "position"],
        var_name="channels",
        value_name="Attention coefficients",
    )
    sns.barplot(
        x="label",
        hue="channels",
        y="Attention coefficients",
        data=m,
        capsize=0.02,
        estimator=np.mean,
        errwidth=1.6,
        ci=95,
    )
    axs.legend(loc="best", bbox_to_anchor=(1, 0.7))

    fig.savefig(
        f"{args.workspace}/channels_attention_coefficients/barplots/{args.seed}.png",
        bbox_inches="tight",
        dpi=300,
    )

    # Load confusion matrix on test set.
    cf_matrix = model.confusion_matrix

    with open(f"{args.workspace}/confusion_matrices_test/{args.seed}.pkl", "wb") as fp:
        pickle.dump(cf_matrix, fp)

    fig, axs = plt.subplots(figsize=(4, 3), ncols=1, nrows=1)
    sns.heatmap(
        cf_matrix / cf_matrix.astype(np.float32).sum(axis=1),
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=axs,
    )
    axs.set_title("Confusion matrix (test)")
    axs.set_ylabel("True label")
    axs.set_xlabel("Predicted label")

    fig.savefig(
        f"{args.workspace}/confusion_matrices_test/heatmaps/{args.seed}.png",
        bbox_inches="tight",
        dpi=300,
    )

    # Get edges topology with GAT attentions.
    edges = model.attentions_topology[0].detach().cpu().numpy()
    att_edges = model.attentions_topology[1].detach().cpu().numpy()

    weight_nodes = []
    for team_id in range(n_teams):
        # Keep nodes that belong to the current team.
        nodes = [node for node, attribute in HeteroData.isolated_graph.nodes(data=True) if attribute["Team"] == team_id]
        # Subgraph.
        subgraph = HeteroData.isolated_graph.subgraph(nodes)
        # Get the edges index position of the current team.
        mask = np.isin(edges.T, np.array(subgraph.edges())).all(axis=1)

        # Get attention edges.
        df = pd.DataFrame(edges.T[mask], columns=["id1", "id2"])
        df["weights"] = att_edges[mask]
        df["team"] = team_id

        if best_params["flow_conv_t"] == "target_to_source":
            group_key = "id2"
        else:
            group_key = "id1"

        df_sum = df.groupby([group_key]).agg(weights=("weights", "sum"), team=("team", "first"))
        # Add information regarding those nodes that not have any information (isolated or not compatible with
        # edge direction).
        df_sum = df_sum.reindex(subgraph.nodes())
        df_sum.set_index("team", append=True, inplace=True)
        df_sum.index.names = ["node", "team"]

        weight_nodes.append(df)

    df_att_topology_nodes = pd.concat(weight_nodes)

    df_att_topology_nodes.to_csv(f"{args.workspace}/topology_attention_coefficients/{args.seed}.csv")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
