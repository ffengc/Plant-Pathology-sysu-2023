import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 混淆矩阵怎么看
# https://zhuanlan.zhihu.com/p/443499860

# exp_number
exp_number = 'exp5'

# 读取CSV文件
data = pd.read_csv(f'/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/test/{exp_number}.txt', dtype=float)
# data = data.to_numpy()
# 提取预测值和真实值
y_pred = data.iloc[:, :6].values
y_true = data.iloc[:, 6:].values

# 计算混淆矩阵
conf_matrix = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
# vlag BrBG_r 后面带_r的就是反过来的颜色 CMRmap GnBu
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='ocean_r', xticklabels=['healthy', 'scab', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew'], yticklabels=['healthy', 'scab', 'frog_eye_leaf_spot', 'rust', 'complex', 'powdery_mildew'])
plt.xlabel('pred')
plt.ylabel('label')
plt.title('conf_matrix')
plt.savefig(f'/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/test/matrix_{exp_number}.png', dpi=1000)