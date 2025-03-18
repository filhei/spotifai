import random
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info, Subset

from spotifai.database.music_database import MusicDatabase
from spotifai.audio.audio_processor import AudioProcessor


DB_PATH = "PATH/TO/music_database.db"


class PlaylistDataset(Dataset):
    """A collection of music (audio) divided into playlists.
    The dataset wraps a (wrapper of a) SQLite database and provides easy access in a machine learning setting.
    Every element in the dataset is a batch of audio patches (short audio clips) that are sampled randomly
    from tracks belonging to a corresponding playlist, and is accessed by the playlist id.
    Due to the non-deterministic behavior, multiple calls to __getitem__ using the same id can yield different results.

    Each batch is created by processing audio tracks (one at a time) from a playlist into patches of equal size,
    until sufficient number of patches have been collected. The patches are then stacked.
    Note that a single track can produce multiple patches, depending on 'time_dimension'.
    See 'AudioProcessor' for a description of the preprocessing step.

    Args:
        database_path (str): path to SQLite database.
        sample_count (int): Number of audio patches per batch.
        feature_dimension (int): Number of mel filterbanks (the number of resulting frequency bins) when producing audio patches.
        time_dimension (int): Number of time bins when producing audio patches.
        multi_processing (bool): If using multiple workers when called from a DataLoader.

    Returns:
        batch (torch.tensor): Stack of audio patches with shape (sample_count, feature_dimension, time_dimension)

    Example:

        >>> ds = PlaylistDataset(
        ...     db_path,
        ...     sample_count=32,
        ...     feature_dimension=64,
        ...     time_dimension=1024,
        ...     multi_processing=False,
        ... )

        >>> x = ds[playlist_id]
        >>> x.shape # (32, 64, 1024)

    """

    def __init__(
        self,
        database_path,
        sample_count=16,
        feature_dimension=80,
        time_dimension=1024,
        multi_processing=False,
    ):
        self.database_path = database_path

        # If called from a DataLoader with multiple workers, the class and its properties will be copied to each worker.
        # To enable multiple connections to the database, each worker must create its own instance of the database class.
        # When 'multi_processing' is True, this is therefore deferred to 'worker_init_fn'
        if multi_processing:
            self.database = None
            self.len = None
        else:
            self.database = MusicDatabase(database_path)
            self.len = self.database.playlist_count

        self.sample_count = sample_count

        # TODO: check if new resample rate of 16.1 khz makes any difference
        self.audio_processor = AudioProcessor(
            n_mels=feature_dimension,
            n_time_bins=time_dimension,
            # max_time_padding=0.25
        )

    def __len__(self):
        if self.len is None:
            return MusicDatabase(
                self.database_path
            ).playlist_count  # TODO: required to delete connection somehow?

        return self.len

    def __getitem__(self, pid):

        if pid >= len(self):
            return None

        # get audio file paths for all tracks in the playlist
        audio_paths = self.database.get_playlist_audio(pid)

        # randomize order
        track_ids = list(audio_paths.keys())
        random.shuffle(track_ids)

        batch = []
        count = 0
        for track_id in track_ids:

            audio_path = audio_paths[track_id]

            # read file and preprocess audio into patches
            audio_patches = self.audio_processor(audio_path)

            if audio_patches is None:
                continue

            # stack patches into a batch
            for patch in audio_patches:
                batch.append(patch)
                count += 1
                if count >= self.sample_count:
                    return torch.stack(batch)

        return None


def worker_init_fn(worker_id):
    """Fascillitates parallel prefetching of batches when using DataLoader.
    Since each worker copies the DataSet and all its properties, including the database connection,
    which is not ideal. A work-around is to delay the database initialization and creating an individual connection
    for each worker, which is done using this worker_init_fn.
    """

    worker_info = torch.utils.data.get_worker_info()

    # if the data has been split into training and validation sets, the dataset will be a subset
    if isinstance(worker_info.dataset, Subset):
        worker_info.dataset.dataset.database = MusicDatabase(
            worker_info.dataset.dataset.database_path
        )
        worker_info.dataset.dataset.len = (
            worker_info.dataset.dataset.database.playlist_count
        )
    else:
        worker_info.dataset.database = MusicDatabase(worker_info.dataset.database_path)
        worker_info.dataset.len = worker_info.dataset.database.playlist_count


def get_playlist_data_loader(
    database_path=DB_PATH,
    sample_count=16,
    feature_dimension=80,
    time_dimension=1024,
    num_workers=0,
    split=None,
):
    """Provides DataLoader(s) that can iterate through PlaylistDataset.
    If split is None, one DataLoader is returned, else two.

    Args:
        database_path (str): path to SQLite database.
        sample_count (int): Number of audio patches per batch.
        feature_dimension (int): Number of mel filterbanks (the number of resulting frequency bins) when producing audio patches.
        time_dimension (int): Number of time bins when producing audio patches.
        num_workers (int): how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        split (float): When splitting the data into training and validation subsets, this is the proportion of elements that go into the first subset.
            Value between 0 and 1. If None, the data is not split and only one DataLoader is returned.

    Returns:
        dataloaders (DataLoader or tuple[DataLoader]): DataLoader(s).
    """
    multi_processing = num_workers > 0

    ds = PlaylistDataset(
        database_path,
        sample_count,
        feature_dimension,
        time_dimension,
        multi_processing=multi_processing,
    )

    prefetch_factor = 4 if num_workers > 0 else None
    worker_init_fn_ = worker_init_fn if multi_processing else None

    # wrap DataSet(s) with DataLoader(s)
    datasets = (ds,) if split is None else split_dataset(ds, split)
    dataloaders = tuple(
        DataLoader(
            dataset,
            batch_size=None,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            worker_init_fn=worker_init_fn_,
        )
        for dataset in datasets
    )
    if len(dataloaders) == 1:
        dataloaders = dataloaders[0]

    return dataloaders


def split_dataset(ds, split):
    """Split a dataset into a partition of two subsets.
    The dataset is split by separating the first (split / len(ds)) share of elements from the remaining ones.

    Args:
        ds (DataSet): The dataset to split.
        split (float): The proportion of elements that go into the first subset.

    Example:
        ds_a, ds_b = split_dataset(ds, 0.9) # (90%, 10%)

    Returns:
        subsets (Subset, Subset): Subsets with lengths: (split * len(ds), (1-split) * len(ds))
    """
    n = len(ds)
    split_point = int(split * n)
    train_indices = range(0, split_point)
    val_indices = range(split_point, n)

    training_dataset = Subset(ds, train_indices)
    validation_dataset = Subset(ds, val_indices)

    return training_dataset, validation_dataset


if __name__ == "__main__":
    dl_a, dl_b = get_playlist_data_loader(DB_PATH, num_workers=2, split=0.9)
