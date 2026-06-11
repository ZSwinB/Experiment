import os
import time
import cv2
import numpy as np
import mitsuba as mi

from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver
)

# =====================================================
# 参数
# =====================================================

SCENE_FILE = r"E:\RM\RM\plane.xml"
scene_id = 0
tx_id = 0

BUILDING_PNG = (
    rf"G:\RadioMapSeer\png\buildings_complete\{scene_id}.png"
)

ANTENNA_PNG = (
    rf"G:\RadioMapSeer\png\antennas\{scene_id}_{tx_id}.png"
)

OUT_DIR = r"E:\RM\RM\SionnRTexperiment"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH = 1

print("Mitsuba variant:", mi.variant())

# =====================================================
# 工具函数
# =====================================================

def to_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.array(x)

# =====================================================
# 读取建筑图
# =====================================================

building = cv2.imread(
    BUILDING_PNG,
    cv2.IMREAD_GRAYSCALE
)

if building is None:
    raise RuntimeError("cannot read building png")

H, W = building.shape

print("map shape:", building.shape)

# =====================================================
# 读取天线图
# =====================================================

ant = cv2.imread(
    ANTENNA_PNG,
    cv2.IMREAD_GRAYSCALE
)

if ant is None:
    raise RuntimeError("cannot read antenna png")

ys, xs = np.where(ant == 255)

if len(xs) == 0:
    raise RuntimeError("cannot find tx")

tx_x = int(xs[0])
tx_y = int(ys[0])

print("tx =", tx_x, tx_y)

# =====================================================
# 输出图
# =====================================================

ratio_map = np.zeros((H, W), dtype=np.float32)
tau_map = np.zeros((H, W), dtype=np.float32)
energy_map = np.zeros((H, W), dtype=np.float32)

# =====================================================
# 场景
# =====================================================

scene = load_scene(
    SCENE_FILE,
    merge_shapes=False
)

scene.frequency = 5.9e9

scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

scene.rx_array = scene.tx_array

# =====================================================
# Tx
# =====================================================

tx = Transmitter(
    name="tx",
    position=[float(tx_x), float(tx_y), 0.5]
)

scene.add(tx)

# =====================================================
# Solver
# =====================================================

solver = PathSolver()

# =====================================================
# 所有空地
# =====================================================

free_space = np.argwhere(building == 0)

coords = []

for y, x in free_space:
    coords.append((int(x), int(y)))

print("free space count:", len(coords))

# =====================================================
# 主循环
# =====================================================

start_time = time.time()

for i in range(0, len(coords), BATCH):

    batch = coords[i:i+BATCH]

    # 清空上一批 Receiver
    scene._receivers = {}

    valid_xy = []

    for j, (x, y) in enumerate(batch):

        rx = Receiver(
            name=f"rx_{j}",
            position=[float(x), float(y), 0.5]
        )

        scene.add(rx)

        valid_xy.append((x, y))

    try:

        paths = solver(
            scene,
            max_depth=2,
            los=True,
            specular_reflection=True,
            diffraction=True,
            edge_diffraction=False,
            refraction=False,
            diffuse_reflection=False
        )

        valid = to_np(paths.valid)
        tau = to_np(paths.tau)
        phi_r = to_np(paths.phi_r)

        a_re, a_im = paths.a

        a_re = to_np(a_re)
        a_im = to_np(a_im)

        if valid.size == 0:

            for x, y in valid_xy:
                ratio_map[y, x] = 0.0
                tau_map[y, x] = 0.0
                energy_map[y, x] = 0.0

            continue
        # -------------------------------------------------
        # reshape
        # -------------------------------------------------

        valid = valid.reshape(-1, valid.shape[-1])
        tau = tau.reshape(-1, tau.shape[-1])
        phi_r = phi_r.reshape(-1, phi_r.shape[-1])

        a_re = a_re.reshape(-1, a_re.shape[-1])
        a_im = a_im.reshape(-1, a_im.shape[-1])

        # -------------------------------------------------
        # 每个 Receiver
        # -------------------------------------------------

        for k, (x, y) in enumerate(valid_xy):

            if k >= valid.shape[0]:
                continue

            v = valid[k]

            if not np.any(v):
                ratio_map[y, x] = 0.0
                tau_map[y, x] = 0.0
                energy_map[y, x] = 0.0
                continue

            power = a_re[k]**2 + a_im[k]**2

            power = power[v]
            tau_k = tau[k][v]
            phi_k = phi_r[k][v]

            if len(power) == 0:

                ratio_map[y, x] = 0.0
                tau_map[y, x] = 0.0
                energy_map[y, x] = 0.0

                continue

            # ---------------------------------------------
            # dominant path
            # ---------------------------------------------

            idx_dom = np.argmax(power)

            phi_dom = phi_k[idx_dom]

            tau_dom = tau_k[idx_dom]

            # ---------------------------------------------
            # sector
            # ---------------------------------------------

            dphi = np.abs(
                np.arctan2(
                    np.sin(phi_k - phi_dom),
                    np.cos(phi_k - phi_dom)
                )
            )

            sector_mask = dphi <= np.deg2rad(60.0)

            energy_total = np.sum(power)

            energy_sector = np.sum(
                power[sector_mask]
            )

            if energy_total <= 0:

                ratio = 0.0

            else:

                ratio = (
                    energy_sector /
                    energy_total
                )

            ratio_map[y, x] = np.float32(ratio)
            tau_map[y, x] = np.float32(tau_dom)
            energy_map[y, x] = np.float32(energy_total)

    except Exception as e:

        print(
            f"batch {i} failed:",
            e
        )

        for x, y in valid_xy:

            ratio_map[y, x] = 0.0
            tau_map[y, x] = 0.0
            energy_map[y, x] = 0.0

    # =================================================
    # 进度
    # =================================================

    if i % (BATCH * 500) == 0:

        elapsed = time.time() - start_time

        print(
            f"{i}/{len(coords)} "
            f"elapsed={elapsed/60:.2f} min"
        )

        np.save(
            os.path.join(
                OUT_DIR,
                f"{scene_id}_{tx_id}_ratio.npy"
            ),
            ratio_map
        )

        np.save(
            os.path.join(
                OUT_DIR,
                f"{scene_id}_{tx_id}_tau.npy"
            ),
            tau_map
        )

        np.save(
            os.path.join(
                OUT_DIR,
                f"{scene_id}_{tx_id}_energy.npy"
            ),
            energy_map
        )

# =====================================================
# 最终保存
# =====================================================

np.save(
    os.path.join(
        OUT_DIR,
        f"{scene_id}_{tx_id}_ratio.npy"
    ),
    ratio_map
)

np.save(
    os.path.join(
        OUT_DIR,
        f"{scene_id}_{tx_id}_tau.npy"
    ),
    tau_map
)

np.save(
    os.path.join(
        OUT_DIR,
        f"{scene_id}_{tx_id}_energy.npy"
    ),
    energy_map
)
elapsed = time.time() - start_time

print("done")
print(f"time = {elapsed/60:.2f} min")
print("saved")