from numpy import float32
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class Resnet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.base = nn.Sequential(
            *list(models.resnet18(pretrained=True).children())[:-2],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten())

        for module in self.base[7][0].modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear1 = nn.Linear(512, 64)
        self.classifer1 = nn.Linear(512, 751)
        self.classifer2 = nn.Linear(64, 751)


    def forward(self, x):
        feature = self.base(x)  # batch_size * 512
        feature1 = self.bn1(feature)
        feature2 = self.bn2(self.linear1(feature1))
        
        if self.training:
            return self.classifer1(feature1), self.classifer2(feature2), feature1, feature2
        else:
            # 模长为1
            feature1 = torch.nn.functional.normalize(feature1)
            feature2 = torch.nn.functional.normalize(feature2)
            return feature1, feature2

class FeatureExtractor:
    def __init__(self, imgSize) -> None:
        self.model = Resnet50()
        self.model.load_state_dict(torch.load('512.pth'))
        self.model.cuda().eval()

        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(imgSize),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    def transforms(self, imgs):
        ''' 将np的数据调整大小后转换成torch '''
        x = []
        for img in imgs:
            img = self.preprocess(img)
            x.append(img)

        x = torch.stack(x, dim=0)
        x = x.to('cuda')
        return x

    @torch.no_grad()
    def __call__(self, imgs):
        x = self.transforms(imgs)
        return self.model(x)
        

if __name__ == '__main__':
    featureExtractor = FeatureExtractor((256, 128), 'cuda')
    import numpy as np
    feature = featureExtractor(np.random.rand(4, 100, 100, 3).astype(np.uint8))[0]
    print(feature.shape)