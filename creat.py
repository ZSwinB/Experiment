import numpy as np
import os

MINOR_PATH = "/root/RM/data/minors/index.npy"
MAJOR_PATH = "/root/RM/data/minors/5_index.npy"

minor_index = np.load(MINOR_PATH)   # (N, 2)

N = minor_index.shape[0]

ratio = 0.05   # 5%
num_sample = int(N * ratio)

np.random.seed(42)   # 可复现

# 随机抽行索引
row_ids = np.random.choice(N, size=num_sample, replace=False)

sampled = minor_index[row_ids]

# 合并（如果已有）
if os.path.exists(MAJOR_PATH):
    major_index = np.load(MAJOR_PATH)
    merged = np.vstack([major_index, sampled])
else:
    merged = sampled

np.save(MAJOR_PATH, merged)

print("original:", N)
print("sampled:", sampled.shape[0])
print("ratio:", sampled.shape[0] / N)