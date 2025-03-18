from spotifai.maching_learning.train import train_model

if __name__ == "__main__":

    experiment_path = "experiments/example"

    train_model(
        experiment_path=experiment_path,
        device="cuda",
        mixed_precision=True,
        multi_processing=True,
    )
