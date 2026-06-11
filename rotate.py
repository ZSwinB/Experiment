import os
import numpy as np
from tqdm import tqdm

# ==========================================
# 参数
# ==========================================

INPUT_DIR = r"/root/RM/pred_dist"
OUTPUT_DIR = r"/root/RM/pred_vector"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SHIFT_DEG = 0.0

# ==========================================
# 旋转矩阵
# ==========================================

theta = np.deg2rad(SHIFT_DEG)

c = np.cos(theta)
s = np.sin(theta)

# ==========================================
# 所有文件
# ==========================================

files = [
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".npy")
]

for fname in tqdm(files):

    path = os.path.join(INPUT_DIR, fname)

    T = np.load(path).astype(np.float32)

    H, W = T.shape

    # ======================================
    # 中心差分
    # ======================================

    dx = np.zeros_like(T)
    dy = np.zeros_like(T)

    dx[:, 1:-1] = (
        T[:, 2:] - T[:, :-2]
    ) * 0.5

    dy[1:-1, :] = (
        T[2:, :] - T[:-2, :]
    ) * 0.5

    # 边界

    dx[:, 0] = T[:, 1] - T[:, 0]
    dx[:, -1] = T[:, -1] - T[:, -2]

    dy[0, :] = T[1, :] - T[0, :]
    dy[-1, :] = T[-1, :] - T[-2, :]

    # ======================================
 
    # ======================================

    vx = dx
    vy = dy

    # ======================================
    # 单位化
    # ======================================

    norm = np.sqrt(
        vx * vx +
        vy * vy
    )

    norm = np.maximum(
        norm,
        1e-8
    )

    vx /= norm
    vy /= norm

    # ======================================
    # 全局旋转
    # ======================================

    vx_rot = (
        c * vx
        - s * vy
    )

    vy_rot = (
        s * vx
        + c * vy
    )

    # ======================================
    # 保存
    # ======================================

    vector_field = np.stack(
        [
            vx_rot,
            vy_rot
        ],
        axis=-1
    ).astype(np.float32)

    save_path = os.path.join(
        OUTPUT_DIR,
        fname
    )

    np.save(
        save_path,
        vector_field
    )

print("done")