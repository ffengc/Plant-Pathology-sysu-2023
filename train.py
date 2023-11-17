import argparse
import math
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import *
from dataset import LeafDataset

# exp
exp_number = 'exp13'

# 命令行传参
# 跑st的时候bsize调成1
# st的bsize只能是1
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='4', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--num_of_epoch', type = int, default=30, help='num of epoch')
parser.add_argument('--batch_size', type = int, default=1, help='batch size')
parser.add_argument('--lr', type = float, default=1e-5, help='learning rate')
parser.add_argument('--train_label_dir', default='/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/data/Train.csv', help='train label dir')
parser.add_argument('--train_image_dir', default='/home/hbenke/Project/Yufc/Project/cv/Data/plant-pathology-data/train/images/', help='train image dir')
parser.add_argument('--val_label_dir', default='/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/data/Val.csv', help='val label dir')
parser.add_argument('--val_image_dir', default='/home/hbenke/Project/Yufc/Project/cv/Data/plant-pathology-data/val/images/', help='val image dir')
parser.add_argument('--save_dir', default=f'/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/run/{exp_number}/', help='save dir')
parser.add_argument('--weights_dir', default='', help='weights path')
parser.add_argument("--log_dir_path", default=f'/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/Log/', help='Log dir')
args = parser.parse_args()

# 超参数
# 设置为gpu训练
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

num_of_epoch = args.num_of_epoch
learning_rate = args.lr
batch_size = args.batch_size

# 数据路径
train_label_dir = args.train_label_dir
train_image_dir = args.train_image_dir
val_label_dir = args.val_label_dir
val_image_dir = args.val_image_dir
save_dir = args.save_dir
weights_dir = args.weights_dir
log_dir = args.log_dir_path

# 创建结果目录
os.system("mkdir -p " + save_dir)

# 打开log文件
train_val_log_f = open(log_dir + f"{exp_number}_train_val.log", 'w')

# 创建Loader
# st的时候改成[384, 384] 原来是[312, 1000] 
train_ds = LeafDataset(csv_file=train_label_dir, imgs_path=train_image_dir,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize([384, 384]),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
                    ]))

val_ds = LeafDataset(csv_file=val_label_dir, imgs_path=val_image_dir,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize([384, 384]),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
                    ]))

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers=4 表示用四个子进程加载数据
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, num_workers=4)

TRAIN_SIZE = len(train_ds)
VALID_SIZE = len(val_ds)

loss_fn = torch.nn.BCEWithLogitsLoss()

print_running_loss = False

# 训练函数
def Train(net, loader):
    tr_loss = 0
    tr_accuracy = 0
    # 创建一个迭代对象
    items = enumerate(loader)
    total_items = len(loader)  # 获取迭代对象的总长度
    for _, (images, labels) in tqdm(items, total=total_items, desc="train"):
        images, labels = images.to(device), labels.to(device)
        # 梯度置0
        optimizer.zero_grad()
        predictions = net(images) # 得到预测值
        loss = loss_fn(predictions, labels.squeeze(-1)) # 是一个 torch 的数字
        net.zero_grad() # 清理梯度
        loss.backward() # 反向传播
        tr_loss += loss.item()
        # 计算准确率
        batch_shape = list(predictions.size())
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                prediction = 1 if predictions.detach().cpu().numpy()[i][j] >= 0.5 else 0
                if prediction == labels.detach().cpu().numpy()[i][j]:
                    tr_accuracy += 1.0/batch_shape[1]
        optimizer.step()
        if print_running_loss and _ % 10 == 0:
            print("One image finished, running loss is" + str(tr_loss/TRAIN_SIZE))
    return tr_accuracy/TRAIN_SIZE, tr_loss/TRAIN_SIZE
# 验证函数
def Eval(net, loader):
    valid_loss = 0
    valid_accuracy = 0 
    with torch.no_grad():          
        # 创建一个迭代对象
        items = enumerate(loader)
        total_items = len(loader)  # 获取迭代对象的总长度    
        for _, (images, labels) in tqdm(items, total=total_items, desc="val"):       
            images, labels = images.to(device), labels.to(device)
            net.eval()
            predictions = net(images)
            loss = loss_fn(predictions, labels.squeeze(-1))       
            valid_loss += loss.item()
            batch_shape = list(predictions.size())
            for i in range(batch_shape[0]):
                for j in range(batch_shape[1]):
                    prediction = 1 if predictions.detach().cpu().numpy()[i][j] >= 0.5 else 0
                    if prediction == labels.detach().cpu().numpy()[i][j]:
                        valid_accuracy += 1.0 / batch_shape[1]
    return valid_accuracy/VALID_SIZE, valid_loss/VALID_SIZE

# 训练
# model_name = "ResNet50 with DE"
# leaf_model = ResNet50(num_classes=6, pretrained=True).to(device)
# model_name = "VGG16 with DE"
# leaf_model = VGG16(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer with DE"
# leaf_model = SwinTransformer(num_classes=6, pretrained=True).to(device)
# model_name = "SwinTransformer_STN_DE"
# leaf_model = SwinTransformer_STN(num_classes=6, pretrained=True).to(device)
model_name = "SwinTransformer_ResNet_ATT_DE"
leaf_model = SwinTransformer_ResNet_ATT(num_classes=6, pretrained=True).to(device)
# model_name = "EfficientNet_DE"
# leaf_model = EfficientNet(num_classes=6, pretrained=True).to(device)
# model_name = "MobileNetV2 with data enhancement"
# leaf_model = MobileNetV2(num_classes=6, pretrained=True).to(device)
# model_name = "Xception"
# leaf_model = Xception(num_classes=6, pretrained=True).to(device)

# checkpoint continue to train
begin_epoch = 0
# leaf_model.load_state_dict(torch.load(f'/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/run/exp4/{begin_epoch}.pt'))

optimizer = optim.Adam(leaf_model.parameters(), lr=learning_rate)

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
train_acc = []
val_acc = []

model_to_save = None # 将要保存的model
max_val_acc = -1

if __name__ == "__main__":
    # 训练
    for epoch in range(begin_epoch, num_of_epoch):
        print(f'Epoch {epoch+1}')
        leaf_model.train() # 设置模型的状态
        ta, tl = Train(leaf_model, loader=train_loader)
        va, vl = Eval(leaf_model, loader=val_loader)
        train_loss.append(tl)
        valid_loss.append(vl)
        train_acc.append(ta)
        valid_acc.append(va)
        print('Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) + ', Train accuracy: ' + str(ta)
            + ', Val loss: ' + str(vl) + ', Val accuracy: ' + str(va))
        train_val_log_f.write('Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) + ', Train accuracy: ' + str(ta)
            + ', Val loss: ' + str(vl) + ', Val accuracy: ' + str(va) + '\n')
        if va >= max_val_acc:
            model_to_save = leaf_model # 存一下当前的模型
            max_val_acc = va
        if epoch % 10 == 0:
            torch.save(leaf_model.state_dict(), save_dir + str(epoch) + ".pt")
            print(f'{str(epoch)}.pt is saved successfully!')
    torch.save(model_to_save.state_dict(), save_dir + "best.pt")
    print('best.pt is saved successfully!')
    torch.save(leaf_model.state_dict(), save_dir + "final.pt")
    print('final.pt is saved successfully!')
    
    # 画图
    epochs = range(1, len(train_loss) + 1) 
    plt.figure(1)
    plt.plot(epochs, train_loss, 'y', label='Training loss')
    plt.plot(epochs, valid_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss' + ' Model: ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir + "loss.jpg", dpi=800)

    plt.figure(2)
    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs, valid_acc, 'b', label='Valid accuracy')
    plt.title('Training and validation accuracy' + ' Model: ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir + "accuracy.jpg", dpi=800)