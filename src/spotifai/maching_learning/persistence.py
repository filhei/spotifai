import csv
import json
import torch
import os
import pandas as pd
import warnings

from spotifai.maching_learning.model import Model


def save_checkpoint(experiment_path, model, optimizer, epoch, step, name=None):
    if experiment_path is None:
        warnings.warn("Cannot save checkpoint since experiment path is None")
        return

    if name is None:
        name = "checkpoint"

    path = os.path.join(experiment_path, f"{name}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # NOTE: if training is stopped mid-epoch, then when resuming, the next epoch will start. However, not really important since the number of steps tracks the training amount.
    epoch = checkpoint["epoch"] + 1
    step = checkpoint["step"]

    return model, optimizer, epoch, step


def save_loss(experiment_path, losses, is_validation=False):
    if experiment_path is None:
        warnings.warn("Cannot save loss since experiment path is None")
        return

    name = "val_history.csv" if is_validation else "history.csv"
    path = os.path.join(experiment_path, name)

    if isinstance(losses, pd.Series):
        losses = losses.copy().items()

    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(losses)


def load_config(experiment_path):
    config_path = os.path.join(experiment_path, "config.json")
    with open(config_path) as f:
        data = json.load(f)

    return data


def load_experiment(experiment_path, checkpoint_path=None):
    # TODO: duplicated code in train.py
    config = load_config(experiment_path)

    model_config = config["model"]
    model = Model(
        model_config["feature_dimension"],
        model_config["time_dimension"],
        model_config["model_dimension"],
        model_config["projection_dimension"],
        layer_count=model_config["layer_count"],
        head_count=model_config["head_count"],
    )

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["learn_rate"])

    model, optimizer, epoch, step = load_checkpoint(
        model,
        optimizer,
        checkpoint_path=checkpoint_path,
    )

    return model, optimizer, config, epoch, step
