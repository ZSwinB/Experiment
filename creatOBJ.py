import os
import cv2
import numpy as np

# ==========================================
# 输入输出
# ==========================================
MAP_ID = 0
IMG_PATH = rf"G:\RadioMapSeer\png\buildings_complete\{MAP_ID}.png"

OUT_DIR = r"E:\RM\RM\obj"
os.makedirs(OUT_DIR, exist_ok=True)

img_id = os.path.splitext(os.path.basename(IMG_PATH))[0]

OUT_OBJ = os.path.join(
    OUT_DIR,
    f"{MAP_ID}.obj"
)

HEIGHT = 1.0

# ==========================================
# 读取图片
# ==========================================

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise RuntimeError(f"Cannot read image: {IMG_PATH}")

H, W = img.shape

print("shape:", img.shape)

# ==========================================
# 255 = wall
# ==========================================

wall_mask = (img == 255)

num_walls = int(np.sum(wall_mask))

print("wall pixels:", num_walls)

# ==========================================
# OBJ生成
# ==========================================

vertex_id = 1

with open(OUT_OBJ, "w") as f:

    for y in range(H):

        for x in range(W):

            if not wall_mask[y, x]:
                continue

            # OBJ坐标
            x0 = float(x)
            x1 = float(x + 1)

            # 为了和图像坐标一致
            y0 = float(y)
            y1 = float(y + 1)

            verts = [

                (x0, y0, 0.0),
                (x1, y0, 0.0),
                (x1, y1, 0.0),
                (x0, y1, 0.0),

                (x0, y0, HEIGHT),
                (x1, y0, HEIGHT),
                (x1, y1, HEIGHT),
                (x0, y1, HEIGHT),
            ]

            for vx, vy, vz in verts:
                f.write(f"v {vx} {vy} {vz}\n")

            v = vertex_id

            # 仅四个侧面
            f.write(f"f {v+0} {v+1} {v+5} {v+4}\n")
            f.write(f"f {v+1} {v+2} {v+6} {v+5}\n")
            f.write(f"f {v+2} {v+3} {v+7} {v+6}\n")
            f.write(f"f {v+3} {v+0} {v+4} {v+7}\n")

            vertex_id += 8

print("saved:", OUT_OBJ)
print("boxes:", num_walls)
print("vertices:", (vertex_id - 1))
print("faces:", num_walls * 4)