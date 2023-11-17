import pandas as pd

# 定义数据路径
data_label_dir = "/home/hbenke/Project/Yufc/Project/cv/Data/plant-pathology-data/"
train_dir = "train/train_label.csv"
test_dir = "test/test_label.csv"
val_dir = "val/val_label.csv"

# 定义项目根目录
project_root_path = '/home/hbenke/Project/Yufc/Project/cv/plant-Pathology-main/'

train_data = pd.read_csv(data_label_dir + train_dir)
test_data = pd.read_csv(data_label_dir + test_dir)
val_data = pd.read_csv(data_label_dir + val_dir)

train_df = pd.DataFrame(columns=["images", "healthy", "scab", "frog_eye_leaf_spot", "rust", 
                                "complex", "powdery_mildew"])
train_df['images'] = train_data['images']
train_df['healthy'] = [0] * len(train_data)
train_df['scab'] = [0] * len(train_data)
train_df['frog_eye_leaf_spot'] = [0] * len(train_data)
train_df['rust'] = [0] * len(train_data)
train_df['complex'] = [0] * len(train_data)
train_df['powdery_mildew'] = [0] * len(train_data)

labels = train_data['labels']
print(labels[0])
for index in range(len(train_data.index)):
    if "healthy" in labels[index]:
        train_df['healthy'][index] = 1
    else:
        if "scab" in labels[index]:
            train_df.at[index, 'scab'] = 1
        if "frog_eye_leaf_spot" in labels[index]:
            train_df.at[index, 'frog_eye_leaf_spot'] = 1
        if "rust" in labels[index]:
            train_df.at[index, 'rust'] = 1
        if "complex" in labels[index]:
            train_df.at[index, 'complex'] = 1
        if "powdery_mildew" in labels[index]:
            train_df.at[index, 'powdery_mildew'] = 1
            
test_df = pd.DataFrame(columns=["images", "healthy", "scab", "frog_eye_leaf_spot", "rust", 
                                "complex", "powdery_mildew"])
test_df['images'] =  test_data['images']
test_df['healthy'] = [0] * len(test_data)
test_df['scab'] = [0] * len(test_data)
test_df['frog_eye_leaf_spot'] = [0] * len(test_data)
test_df['rust'] = [0] * len(test_data)
test_df['complex'] = [0] * len(test_data)
test_df['powdery_mildew'] = [0] * len(test_data)

labels = test_data['labels']
print(labels[0])
for index in range(len(test_data.index)):
    if "healthy" in labels[index]:
        test_df['healthy'][index] = 1
    else:
        if "scab" in labels[index]:
            test_df.at[index, 'scab'] = 1
        if "frog_eye_leaf_spot" in labels[index]:
            test_df.at[index, 'frog_eye_leaf_spot'] = 1
        if "rust" in labels[index]:
            test_df.at[index, 'rust'] = 1
        if "complex" in labels[index]:
            test_df.at[index, 'complex'] = 1
        if "powdery_mildew" in labels[index]:
            test_df.at[index, 'powdery_mildew'] = 1

val_df = pd.DataFrame(columns=["images", "healthy", "scab", "frog_eye_leaf_spot", "rust", 
                                "complex", "powdery_mildew"])
val_df['images'] = val_data['images']
val_df['healthy'] = [0] * len(val_data)
val_df['scab'] = [0] * len(val_data)
val_df['frog_eye_leaf_spot'] = [0] * len(val_data)
val_df['rust'] = [0] * len(val_data)
val_df['complex'] = [0] * len(val_data)
val_df['powdery_mildew'] = [0] * len(val_data)

labels = val_data['labels']
print(labels[0])
for index in range(len(val_data.index)):
    if "healthy" in labels[index]:
        val_df['healthy'][index] = 1
    else:
        if "scab" in labels[index]:
            val_df.at[index, 'scab'] = 1
        if "frog_eye_leaf_spot" in labels[index]:
            val_df.at[index, 'frog_eye_leaf_spot'] = 1
        if "rust" in labels[index]:
            val_df.at[index, 'rust'] = 1
        if "complex" in labels[index]:
            val_df.at[index, 'complex'] = 1
        if "powdery_mildew" in labels[index]:
            val_df.at[index, 'powdery_mildew'] = 1


# 输出csv
train_df.to_csv(project_root_path + "./data/Train.csv", index=False)
test_df.to_csv(project_root_path + "./data/Test.csv", index=False)
val_df.to_csv(project_root_path + "./data/Val.csv", index=False)