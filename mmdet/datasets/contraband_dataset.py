from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class contrabandDataset(CocoDataset):
    CLASSES = ('gun', 'dagger', 'knife', 'bottle', 'scissors')
