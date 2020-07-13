import sys
sys.path.append('../')

from baseline import My_Res50, My_Attention_Net_Gated
import torch
import torchvision.transforms as transforms
from PIL import Image


class Online_Classificaton_Model():
    def __init__(self, model_name, checkpoint_path):
        super(Online_Classificaton_Model, self).__init__()

        self.model_name = model_name.split('/')[-1]
        self.checkpoint_path = checkpoint_path

        self.classes = ['battery', 'bottle', 'firecracker', 'grenade', 'gun', 'hammer', 'knife', 'scissors']
        self.thre = torch.Tensor([0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.7, 0.5])
        self.num_classes = len(self.classes)

        self.transform = self.init_transformers()
        self.model = self.init_models()

    def init_models(self):
        model = eval(self.model_name)(num_classes=self.num_classes)
        if torch.cuda.is_available():
            model = model.cuda()
            checkpoint = torch.load(self.checkpoint_path, map_location='cuda')
            self.thre = self.thre.cuda()

        else:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])
        return model.eval()

    def init_transformers(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        return transform

    def loader(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def predict(self, img_path):
        predictions = list()

        sample = self.loader(img_path)
        sample = self.transform(sample)
        sample = sample.view(1, sample.shape[0], sample.shape[1], sample.shape[2])
        if torch.cuda.is_available():
            sample = sample.cuda()

        logit = self.model(sample, 0)
        pred = torch.sigmoid(logit)[0, :]
        pred = pred > self.thre

        for i in range(self.num_classes):
            if pred[i]:
                predictions.append(self.classes[i])

        return predictions
