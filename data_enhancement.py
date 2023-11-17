import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import RandAugment

def enhance(img):
    img_shape = img.shape
    # 定义图像预处理和增强的转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        RandAugment(N=2, M=9),  # N is the number of augmentation transformations to apply sequentially, M is the magnitude for all the transformations.
        transforms.Resize((img_shape[1], img_shape[2]), interpolation=InterpolationMode.BILINEAR),  # resize back to original size
        transforms.ToTensor(),
    ])
    # 应用数据增强
    augmented_image = transform(img)
    # 现在augmented_image是一个增强后的图像，它仍然是一个[3, 384, 384]的torch.Tensor
    return augmented_image
