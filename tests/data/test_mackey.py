import pytest

from torsk.data import MackeyDataset
from torsk.data.mackey import _simulate_mackey


def test_mackey():

    mackey = _simulate_mackey(N=100)
    assert mackey.shape == (100,)

    with pytest.raises(ValueError):
        dataset = MackeyDataset(100, 100, 200)

    dataset = MackeyDataset(100, 100, 300)
    assert len(dataset) == 100

    inputs, labels, pred_labels, _ = dataset[0]
    assert inputs.shape == (100, 1)
    assert labels.shape == (100, 1)
    assert pred_labels.shape == (100, 1)

    with pytest.raises(IndexError):
        inputs, labels = dataset[199]

    with pytest.raises(IndexError):
        inputs, labels = dataset[-1]
