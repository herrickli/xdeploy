from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class VOCDataset(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')


    CLASSES = ('umbrella', 'liquid', 'lighter', 'scissors', 'knife',
               'cellphone', 'battery', 'gun', 'none-liquid', 'pliers', 'wrench')
    CLASSES = ('gun', 'dagger', 'knife', 'bottle', 'light_bottle', 'scissors')

    CLASSES = ('hammer',  'scissors', 'knife', 'bottle', 'battery', 'firecracker',
                'gun', 'grenade', 'bullet', 'lighter', 'ppball', 'baton')

    CLASSES = ('hammer', 'scissors', 'knife', 'bottle', 'battery', 'firecracker',
               'gun', 'grenade', 'bullet')

    CLASSES = ('gun', 'knife', 'bottle', 'scissors', 'dagger')

    CLASSES = ('hammer','scissors','knife','bottle','battery','firecracker',
                'gun','grenade','bullet','lighter','ppball','baton')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
