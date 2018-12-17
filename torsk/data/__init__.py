from torsk.data.mackey import MackeyDataset
from torsk.data.ocean import NetcdfDataset, DCTNetcdfDataset, SCTNetcdfDataset
from torsk.data.sine import SineDataset
from torsk.data.circle import CircleDataset

__all__ = [
    "NetcdfDataset", "DCTNetcdfDataset", "MackeyDataset", "SineDataset",
    "CircleDataset", "SCTNetcdfDataset"]
