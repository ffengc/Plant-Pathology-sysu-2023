import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=pretrained)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = models.vgg16(pretrained=pretrained)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
    def forward(self, x):
        x = self.model(x)
        return x
    
# swin transformer
import timm
class SwinTransformer(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class EfficientNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    def forward(self, x):
        x = self.model(x)
        return x

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    

class Xception(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model = models.xception(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x