

##### 处理few-shot的代码####
import pickle
import numpy as np

train_path = '/home/chentiankai987/Code/GSP/few-shot/SR/scanobjectnn_train15_test_2048pts_fps.dat'
test_path  = '/home/chentiankai987/Code/GSP/few-shot/SR/scanobjectnn15_test_2048pts_fps.dat'
out_path   = '/home/chentiankai987/Code/GSP/few-shot/SR/scanobjectnn15_test_2048pts_fps_8shot.dat'

# 想从每个类别选多少个样本，在这里改
n_per_class = 8   # 比如每类选 5 个，自己改数字就行

np.random.seed(42)  # 可选：为了可复现

# 1. 读取训练集
with open(train_path, 'rb') as f:
    train_data = pickle.load(f)

x_train = np.array(train_data[0])      # (N_train, P, C)
y_train = np.array(train_data[1])      # (N_train, 1) 或 (N_train,)
y_train_flat = y_train.reshape(-1)

# 2. 读取测试集
with open(test_path, 'rb') as f:
    test_data = pickle.load(f)

x_test = np.array(test_data[0])        # (N_test, P, C)
y_test = np.array(test_data[1])
y_test_flat = y_test.reshape(-1)

print("Train x shape:", x_train.shape)
print("Train y shape:", y_train.shape)
print("Test  x shape:", x_test.shape)
print("Test  y shape:", y_test.shape)

# 3. 从训练集中每个类别随机选 n_per_class 个样本
unique_labels = np.unique(y_train_flat)
print("number of classes in train:", len(unique_labels))
print("class indices in train:", unique_labels)

selected_indices_list = []

for cls in unique_labels:
    cls_indices = np.where(y_train_flat == cls)[0]
    if len(cls_indices) < n_per_class:
        raise ValueError(
            f"Class {cls} only has {len(cls_indices)} samples, "
            f"but you asked for n_per_class={n_per_class}."
        )
    chosen = np.random.choice(cls_indices, n_per_class, replace=False)
    selected_indices_list.append(chosen)

# 拼成一维索引
selected_indices = np.concatenate(selected_indices_list, axis=0)  # (num_classes * n_per_class,)

# 取出对应的点云和标签
x_from_train = x_train[selected_indices]          # (num_classes * n_per_class, P, C)
y_from_train = y_train_flat[selected_indices]     # (num_classes * n_per_class,)

# 4. 改成：先测试集，再把这些训练样本拼在“后面”
x_new = np.concatenate([x_test, x_from_train], axis=0)

y_new = np.concatenate([y_test_flat, y_from_train], axis=0)
y_new = y_new.reshape(-1, 1)  # 保持和原 .dat 一样的 (N, 1) 形状

print("New test x shape:", x_new.shape)
print("New test y shape:", y_new.shape)

# 5. 保存新的测试集
new_data = [x_new, y_new]

with open(out_path, 'wb') as f:
    pickle.dump(new_data, f)

print("Saved new test set with extra samples to:", out_path)
print("Added samples (from train):", len(unique_labels) * n_per_class)



##### 处理full-shot的代码####



# import pickle
# import numpy as np

# train_path = '/home/chentiankai987/Code/GSP/few-shot/SR/scanobjectnn_train15_test_2048pts_fps.dat'
# test_path  = '/home/chentiankai987/Code/GSP/few-shot/SR/scanobjectnn15_test_2048pts_fps.dat'
# out_path   = '/home/chentiankai987/Code/GSP/few-shot/SR/scanobjectnn15_test_2048pts_fps_fullshot.dat'

# # 1. 读取训练集
# with open(train_path, 'rb') as f:
#     train_data = pickle.load(f)

# x_train = np.array(train_data[0])      # (N_train, P, C)
# y_train = np.array(train_data[1])      # (N_train, 1) 或 (N_train,)
# y_train_flat = y_train.reshape(-1)     # 拉平方便拼接

# # 2. 读取测试集
# with open(test_path, 'rb') as f:
#     test_data = pickle.load(f)

# x_test = np.array(test_data[0])        # (N_test, P, C)
# y_test = np.array(test_data[1])        # (N_test, 1) 或 (N_test,)
# y_test_flat = y_test.reshape(-1)

# print("Train x shape:", x_train.shape)
# print("Train y shape:", y_train.shape)
# print("Test  x shape:", x_test.shape)
# print("Test  y shape:", y_test.shape)

# # 3. 测试集在前，训练集在后
# x_new = np.concatenate([x_test, x_train], axis=0)               # (N_test + N_train, P, C)
# y_new_flat = np.concatenate([y_test_flat, y_train_flat], axis=0)  # (N_test + N_train,)

# # 标签恢复成 (N, 1) 形状，和原 .dat 格式一致
# y_new = y_new_flat.reshape(-1, 1)

# print("New x shape (test first, then train):", x_new.shape)
# print("New y shape:", y_new.shape)

# # 4. 打包并保存
# new_data = [x_new, y_new]

# with open(out_path, 'wb') as f:
#     pickle.dump(new_data, f)

# print("Saved new merged set (test first, then train) to:", out_path)
