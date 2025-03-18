import json
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
import os
import torch
from tqdm import tqdm

from spotifai.maching_learning.buffer import Buffer
from spotifai.maching_learning.loss import ContrastiveLoss, similarity_groups
from spotifai.maching_learning.model import Model
from spotifai.maching_learning.persistence import (
    load_config,
    load_checkpoint,
    save_checkpoint,
    save_loss,
)
from spotifai.maching_learning.playlist_dataset import get_playlist_data_loader

DB_PATH = "PATH/TO/music_database.db"


def parameters_to_config(
    # model
    feature_dimension=96,
    time_dimension=1024,
    model_dimension=256,
    projection_dimension=512,
    layer_count=8,
    head_count=8,
    # optimizer
    learn_rate=3e-4,
    # training
    epochs=2,
    batch_size=32,
    buffer_size=24,
    buffer_update_frequency=2,
    val_period=1000,
    mixed_precision=True,
    # data
    data_split=0.9,
    database_path=DB_PATH,
    # progress
    save_period=5000,
    checkpoint_period=50000,
    loss_aggregation_window=1000,
):
    """Setup a configuration in nested dictionary format from the expected parameters.

    Args:
        feature_dimension (int): Feature dimension of input data (number of frequency bins or mels).
        time_dimension (int): Time dimension of input data (number of time bins).
        model_dimension (int): Hidden size of model layers.
        projection_dimension (int): Output size of projection layer.
        layer_count (int): Number of transformer layers.
        head_count (int): Number of attention heads in each transformer layer.
        learn_rate (float): Learning rate.
        epochs (int): Training duration.
        batch_size (int): Batch size during training.
        buffer_size (int): Number of batches the buffer can store.
        buffer_update_frequency (int): How often new data should be added to the buffer per training step. A frequency of 2 means that two new batches are added to the buffer each backprop step.
        val_period (int): How often a new validation loss should be calculated. A period of 1000 means that every 1000 steps of backprop, a validation epoch is performed.
        mixed_precision (bool): Whether to use half precision (FP16).
        data_split (float): How much of the data should be used for training and validation respectively. A value of 0.9 means that 90% of the data is used for training and 10% for validation.
        database_path (str): Path to SQLite database containing playlist and tracks data.
        save_period (int): How often training (in terms of backprop steps) logs and a 'latest model checkpoint' should be saved. This model checkpoint is always overwritten to only store the latest weights.
        checkpoint_period (int): How often a historical checkpoint should be saved. These checkpoints will persist and not be overwritten, allowing probing of and comparing models with different amounts of training.
        loss_aggregation_window (int): The amount of backprop steps over which the loss is aggregated. A window of size 1000 means that the loss is averaged over and saved every 1000 steps.

    Returns:
        config (dict): Nested dictionary of experiment specific parameters.
    """
    config = {
        "model": {
            "feature_dimension": feature_dimension,
            "time_dimension": time_dimension,
            "model_dimension": model_dimension,
            "projection_dimension": projection_dimension,
            "layer_count": layer_count,
            "head_count": head_count,
        },
        "optimizer": {
            "learn_rate": learn_rate,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "buffer_update_frequency": buffer_update_frequency,
            "val_period": val_period,
        },
        "data": {
            "database_path": database_path,
            "data_split": data_split,
        },
        "progress": {
            "save_period": save_period,
            "checkpoint_period": checkpoint_period,
            "loss_aggregation_window": loss_aggregation_window,
        },
    }
    return config


def train_model(
    experiment_path=None,
    device="cuda",
    mixed_precision=True,
    multi_processing=True,
    config=None,
):
    """Start or continue model training. Each model is associated to a unique experiment, which is contained in a directory ('experiment_path').
    If config is None, an existing experiment is loaded from 'experiment_path' and training is continued.
    If config is a dictionary, a new experiment is created and the training of a new model is started.
    The experiments folder will collect the configuration, model checkpoints and training and validation history (losses).

    Args:
        experiment_path (str): Path to experiment directory.
        device (torch.device): Torch device.
        mixed_precision (bool): Whether to use half precision (FP16).
        multi_processing (bool): Whether to use multiple workers to precompute and preparing data batches.
        config (dict): Nested dictionary of experiment specific parameters (see the 'parameters_to_config' method).
    """

    # setup core functions
    model, optimizer, config, starting_epoch, step = _setup_experiment(
        experiment_path, config, device
    )
    train_dataloader, val_dataloader = _setup_data(config, multi_processing)
    loss_function = ContrastiveLoss().to(device)

    # dependent parameters
    val_steps = (1 - config["data"]["data_split"]) * config["training"]["val_period"]
    half_batch_size = config["training"]["batch_size"] // 2
    dtype = (
        torch.half if mixed_precision else torch.float
    )  # TODO: should half be bfloat16 instead? some issues with numerical instability with vector products using @-operator
    groups = similarity_groups(half_batch_size, config["training"]["buffer_size"])

    # data structures
    data_shape = (
        config["training"]["batch_size"],
        config["model"]["feature_dimension"],
        config["model"]["time_dimension"],
    )
    train_buffer = Buffer(
        size=config["training"]["buffer_size"],
        data_shape=data_shape,
        device=device,
        dtype=dtype,
    )
    val_buffer = Buffer(
        size=config["training"]["buffer_size"],
        data_shape=data_shape,
        device="cpu",
        dtype=dtype,
    )

    # mixed precision utils
    scaler = torch.GradScaler()

    for epoch in range(starting_epoch, config["training"]["epochs"]):
        step = train_epoch(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            scaler=scaler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_buffer=train_buffer,
            val_buffer=val_buffer,
            groups=groups,
            batch_size=config["training"]["batch_size"],
            device=device,
            mixed_precision=mixed_precision,
            val_period=config["training"]["val_period"],
            buffer_update_frequency=config["training"]["buffer_update_frequency"],
            save_period=config["progress"]["save_period"],
            checkpoint_period=config["progress"]["checkpoint_period"],
            loss_aggregation_window=config["progress"]["loss_aggregation_window"],
            experiment_path=experiment_path,
            step=step,
            epoch=epoch,
            val_steps=val_steps,
        )

    torch.cuda.empty_cache()

    return model


def _setup_experiment(experiment_path, config, device):
    assert experiment_path is not None, "'experiment_path' must not be None."

    # load existing experiment
    if config is None:
        model, optimizer, config, epoch, step = _load_experiment(
            experiment_path, device=device
        )

    # create new experiment
    else:
        model, optimizer = _create_experiment(config, experiment_path)
        epoch, step = 0, 0

    return model, optimizer, config, epoch, step


def _load_experiment(experiment_path, checkpoint_path=None, device="cpu"):

    if checkpoint_path is None:
        checkpoint_path = os.path.join(experiment_path, "checkpoint.pt")

    # TODO: raise if cannot load config
    config = load_config(experiment_path)
    model, optimizer = _setup_model_and_optimizer(config, device)

    model, optimizer, epoch, step = load_checkpoint(
        model,
        optimizer,
        checkpoint_path,
    )

    return model, optimizer, config, epoch, step


def _setup_model_and_optimizer(config, device):
    """Combined setup of model and optimizer since the model parameters already need to be on the correct device when initializing the optimizer."""

    model_config = config["model"]
    model = Model(
        model_config["feature_dimension"],
        model_config["time_dimension"],
        model_config["model_dimension"],
        model_config["projection_dimension"],
        layer_count=model_config["layer_count"],
        head_count=model_config["head_count"],
    )
    model.to(device)

    optimizer_config = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["learn_rate"])

    return model, optimizer


def _create_experiment(config, experiment_path=None):
    model, optimizer = _setup_model_and_optimizer(config)

    if experiment_path is not None:
        path = Path(experiment_path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w") as file:
            json.dump(config, file, indent=4)

    return model, optimizer


def _setup_data(config, multi_processing):
    worker_count = multiprocessing.cpu_count() if multi_processing else 0
    train_dataloader, val_dataloader = get_playlist_data_loader(
        db_path=config["data"]["database_path"],
        sample_count=config["training"]["batch_size"],
        feature_dimension=config["model"]["feature_dimension"],
        time_dimension=config["model"]["time_dimension"],
        num_workers=worker_count,
        split=config["data"]["data_split"],
    )

    return train_dataloader, val_dataloader


def train_epoch(
    model=None,
    optimizer=None,
    loss_function=None,
    scaler=None,
    train_dataloader=None,
    val_dataloader=None,
    train_buffer=None,
    val_buffer=None,
    groups=None,
    batch_size=None,
    device=None,
    mixed_precision=None,
    val_period=None,
    buffer_update_frequency=None,
    save_period=None,
    checkpoint_period=None,
    loss_aggregation_window=None,
    experiment_path=None,
    step=None,
    epoch=None,
    val_steps=None,
):
    """Train one epoch worth of model updates.

    Args:
        model (torch.nn.Module): Neural network.
        optimizer (torch.optim.Optimizer): Pytorch optimizer class.
        loss_function (torch.nn.Module): Contrastive loss function.
        scaler (torch.GradScaler): Scaler for mixed precision calculations.
        train_dataloader (torch.utils.data.DataLoader): Training dataloader.
        val_dataloader (torch.utils.data.DataLoader): Validation dataloader.
        train_buffer (Buffer): Buffer for training data.
        val_buffer (Buffer): Buffer for validation data.
        groups (list[list[int]]): List of list of indices of observations from the corresponding contrastive class.
        batch_size (int): Batch size.
        device (torch.device): Torch device.
        mixed_precision (bool): Whether to use half precision (FP16).
        val_period (int): How often a new validation loss should be calculated. A period of 1000 means that every 1000 steps of backprop, a validation epoch is performed.
        buffer_update_frequency (int): How often new data should be added to the buffer per training step. A frequency of 2 means that two new batches are added to the buffer each backprop step.
        save_period (int): How often training (in terms of backprop steps) logs and a 'latest model checkpoint' should be saved. This model checkpoint is always overwritten to only store the latest weights.
        checkpoint_period (int): How often a historical checkpoint should be saved. These checkpoints will persist and not be overwritten, allowing probing of and comparing models with different amounts of training.
        loss_aggregation_window (int): The amount of backprop steps over which the loss is aggregated. A window of size 1000 means that the loss is averaged over and saved every 1000 steps.
        experiment_path (_type_, optional): _description_. Defaults to None.
        step (int): Current training step.
        epoch (int): Current epoch.
        val_steps (int): Number of steps (forward passes) per validation epoch.

    Returns:
        step (int): Next training step.
    """
    model.train()  # unfreeze trainable parameters

    iteration = 0
    losses = []
    mean_val_loss = np.nan

    progress_bar = tqdm(train_dataloader, leave=True, desc=f"Epoch: {epoch}")

    for batch in progress_bar:

        # add one batch of data to the buffer
        buffer_is_full_and_has_new_data = _validate_and_buffer_data(
            train_buffer, batch, batch_size
        )
        if not buffer_is_full_and_has_new_data:
            continue

        # only train every n buffer updates
        if iteration % buffer_update_frequency != 0:
            iteration += 1
            continue

        # training step
        optimizer.zero_grad()

        loss = _forward_pass(
            model, train_buffer, loss_function, groups, device, mixed_precision
        )
        # TODO: remove debugging lines
        # if torch.isinf(loss):
        #     print(torch.isinf(train_buffer.data).any())

        losses.append((step, loss.item()))

        _backward_pass(loss, optimizer, scaler)

        iteration += 1
        step += 1

        # validation step
        if step % val_period == 0:
            # leave room on gpu (should not be a problem, but who knows)
            train_buffer.to("cpu", empty_cuda_cache=True)
            val_buffer.to(device)

            # TODO: freeze batch norm parameters [NOTE: model currently relies on BN to normalize outputs - this should be done manually, but until then, eval() should not be called]
            # model.eval()

            mean_val_loss = _validation_round(
                model,
                val_dataloader,
                loss_function,
                val_buffer,
                batch_size,
                groups,
                device,
                mixed_precision,
                val_steps,
                progress_bar,
                step,
            )

            save_loss(
                experiment_path,
                [(step, mean_val_loss)],
                is_validation=True,
            )

            # TODO: uncomment [see above NOTE]
            # model.train()
            val_buffer.to("cpu", empty_cuda_cache=True)
            train_buffer.to(device)

        # diagnostics
        rolling_mean_loss = np.mean([l for s, l in losses[-32:]])
        progress_bar.set_postfix(
            {
                "step": step,
                "loss": f"{rolling_mean_loss: 8.4f}",
                "val loss": f"{mean_val_loss: 8.4f}",
            }
        )

        # save results and update most recent checkpoint
        if step % save_period == 0:

            _adjust_and_save_training_loss(
                losses, loss_aggregation_window, experiment_path
            )
            save_checkpoint(experiment_path, model, optimizer, epoch, step)

        # save historical checkpoint
        if step % checkpoint_period == 0:
            suffix = step // 1000
            save_checkpoint(
                experiment_path,
                model,
                optimizer,
                epoch,
                step,
                name=f"checkpoint_{suffix}k",
            )

    # save final results and checkpoint for epoch
    _adjust_and_save_training_loss(losses, loss_aggregation_window, experiment_path)
    save_checkpoint(experiment_path, model, optimizer, epoch, step)

    return step


def _is_valid_batch(batch, batch_size):
    # some playlists do not contain enough audio to analyze

    if batch is None:
        return False

    if len(batch) < batch_size:
        return False

    return True


def _validate_and_buffer_data(buffer, batch, expected_batch_size):
    """
    Returns:
        buffer_is_full_and_has_new_data (bool): whether buffer is filled and has new data.
    """
    if not _is_valid_batch(batch, expected_batch_size):
        return False

    # buffer batches
    buffer.add(batch)

    if not buffer.has_data:
        return False

    return True


def _forward_pass(model, train_buffer, loss_function, groups, device, mixed_precision):

    # chunk before inference instead of after, since batch normalization needs to be applied per "view"
    x_a, x_b = train_buffer.data.chunk(chunks=2, dim=0)

    with torch.autocast(device_type=device, enabled=mixed_precision):
        z_a = model(x_a, projection=True)
        z_b = model(x_b, projection=True)

        loss = loss_function(z_a, z_b, groups)

    return loss


def _backward_pass(loss, optimizer, scaler):
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


@torch.no_grad()
def _validation_round(
    model,
    val_dataloader,
    loss_function,
    val_buffer,
    batch_size,
    groups,
    device,
    mixed_precision,
    val_steps,
    progress_bar=None,
    step=None,
):
    val_losses = []
    val_step = 0

    for val_batch in val_dataloader:

        # print status
        if progress_bar is not None:
            progress_bar.set_postfix(
                {"step": step, "validation": f"{val_step / val_steps:.2%}"}
            )

        # add one batch of data to the buffer
        buffer_is_full_and_has_new_data = _validate_and_buffer_data(
            val_buffer, val_batch, batch_size
        )
        if not buffer_is_full_and_has_new_data:
            continue

        # validation step
        val_loss = _forward_pass(
            model,
            val_buffer,
            loss_function,
            groups,
            device,
            mixed_precision,
        )
        # TODO: remove debugging lines
        # if torch.isinf(val_loss):
        #     print("Val loss is inf")
        #     print(torch.isinf(val_buffer.data).any())

        val_losses.append(val_loss.item())

        val_step += 1
        if val_step >= val_steps:
            break

    mean_val_loss = np.mean(val_losses)

    return mean_val_loss


def _adjust_and_save_training_loss(losses, loss_aggregation_window, experiment_path):
    """Aggregate losses by reducing over n steps (mean of 'loss_aggregation_window')."""
    losses_df = pd.DataFrame(losses, columns=["step", "loss"])
    average_losses = losses_df.groupby(losses_df.step // loss_aggregation_window)[
        "loss"
    ].mean()
    average_losses.index = (average_losses.index + 1) * loss_aggregation_window

    remainder = len(losses) % loss_aggregation_window
    if remainder != 0:
        # only save 'complete' aggregations
        average_losses = average_losses.iloc[:-1]

        # keep unused losses until next time
        losses = losses[-remainder:]
    else:
        losses.clear()

    save_loss(experiment_path, average_losses)
