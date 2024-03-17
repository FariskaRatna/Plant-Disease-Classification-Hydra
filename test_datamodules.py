from pathlib import Path

import pytest
import torch

from src.data.plant_datamodule import PlantEfficientDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_plant_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = PlantEfficientDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.perform_data_split()
    dm.setup()

    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    # assert num_datapoints == 70_000

    num_train_datapoints = len(dm.train_dataset)
    num_val_datapoints = len(dm.val_dataset)
    num_test_datapoints = len(dm.test_dataset)
    total_datapoints = num_train_datapoints + num_val_datapoints + num_test_datapoints

    assert total_datapoints > 0

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

if __name__ == "__main__":
    pytest.main([__file__])
