from .voc import VOCDataset
from .registry import DATASETS

@DATASETS.register_module
class VOCcontrabandDataset(VOCDataset):
    CLASSES = ('gun', 'dagger', 'knife', 'bottle', 'light_bottle', 'scissors')
