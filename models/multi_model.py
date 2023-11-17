import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import ViT
import torch.nn.functional as F
import torch.nn.functional as F
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 直接添加注意力机制，直接用jack的就行了
class AttentionModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh(),
            nn.Linear(out_features, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        weights = self.attention(x)
        return (x * weights).sum(dim=1)

class SwinTransformer_ResNet_ATT(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.model1 = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)
        self.model2 = timm.create_model('resnet50', pretrained=pretrained)
        self.model1.head = nn.Linear(self.model1.head.in_features, num_classes)
        self.model2.fc = nn.Linear(self.model2.fc.in_features, num_classes)
        self.attention = AttentionModule(num_classes, num_classes)
        # 这里的fc感觉还要再设计一下，感觉有点垃圾
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = torch.stack([x1, x2], dim=1)
        x = self.attention(x)
        return x
    
'''
因为resnet和st的性能比较好，所以直接搞一下ebend的
两个模型st和resnet50
并且加入了一个注意力模块来学习这两个模型输出的权重。
这样，模型可以根据输入的特点，自动决定更加依赖哪个模型的输出。
'''