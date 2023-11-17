import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import ViT
import torch.nn.functional as F
import torch.nn.functional as F
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 升级版本的 SwinTransformer 添加了Dropout和一些Liner
# 这个随便加的，比较垃圾
class SwinTransformer_advanced(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.5):
        super().__init__()
        self.model = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(self.model.head.in_features)
        self.fc1 = nn.Linear(self.model.head.in_features, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# 这个很常用，但是感觉效果不会好
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(84640, 32), # 这个大小要对上面的大小
            nn.ReLU(True), 
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    def forward(self, x):
        xs = self.localization(x)
        # 自己调整添加的
        x_size_lst = list(xs.size())
        total_size = x_size_lst[0] * x_size_lst[1] * x_size_lst[2] * x_size_lst[3]
        xs = xs.view(x_size_lst[0], total_size // x_size_lst[0]) # 这里是自己调整的
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class SwinTransformer_STN(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.stn = STN()
        self.model = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)
        self.encoder_layers = TransformerEncoderLayer(d_model=self.model.num_features, nhead=4)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers=2)
        self.fc = nn.Linear(self.model.num_features, num_classes)  
    def forward(self, x):
        x = self.stn(x)
        x = self.model(x)
        x = x.unsqueeze(1)
        x = F.interpolate(x, size=1536, mode='linear', align_corners=False) # 加入一个插值
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x
    
    
'''
这个模型首先使用了一个空间变换网络（STN）模块，
该模块可以学习输入图像的空间变换参数，以增强模型对图像变换

（如旋转、缩放、剪裁等）的鲁棒性。然后，我们使用了st模型进行特征提取
，但移除了原始的分类头，以便我们可以添加自己的模块。最后，

直接tmd添加了一个基于自注意力机制的Transformer编码器，它可以帮助模型更好地理解图像中的全局依赖关系，
然后通过一个全连接层进行分类。
'''

