from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# ==================================================
# 1. 读取 MAT 文件
# ==================================================

folder_path = Path(__file__).resolve().parent
file_path = folder_path / "PIV3000-clean.mat"

data = sio.loadmat(file_path)


# ==================================================
# 2. 取出坐标和所有速度数据
# ==================================================

x = data["x"][0, 0]
y = data["y"][0, 0]

u_cells = data["u"].squeeze()
v_cells = data["v"].squeeze()

# 合并后的尺寸为：
# (2999, 53, 123)
#   帧数   行   列
u_all = np.stack(u_cells, axis=0)
v_all = np.stack(v_cells, axis=0)

total_frames = u_all.shape[0]

print("总帧数：", total_frames)


# ==================================================
# 3. 设置样本数量和剖面位置
# ==================================================

sample_counts = [500, 1000, 2000, total_frames]

# 选择第60列
# 人习惯从第1列开始数，Python从0开始数
column_number = 60
column_index = column_number - 1

# 第60列上53个点的y坐标
distance = y[:, column_index]


# ==================================================
# 4. 计算不同样本数量的平均速度剖面
# ==================================================

profiles = {}

for sample_number in sample_counts:

    # 取前500帧、前1000帧……
    u_sample = u_all[:sample_number]
    v_sample = v_all[:sample_number]

    # 对这些帧求平均
    u_mean = np.mean(u_sample, axis=0)
    v_mean = np.mean(v_sample, axis=0)

    # 计算平均速度大小
    mean_speed = np.sqrt(
        u_mean**2 + v_mean**2
    )

    # 取第60列的竖直剖面
    speed_profile = mean_speed[:, column_index]

    # 保存这一条剖面
    profiles[sample_number] = speed_profile


# ==================================================
# 5. 与2999帧剖面比较误差
# ==================================================

reference_profile = profiles[total_frames]

print("\n样本数量    相对误差")

for sample_number in sample_counts:

    current_profile = profiles[sample_number]

    difference = np.sqrt(
        np.mean(
            (current_profile - reference_profile) ** 2
        )
    )

    reference_size = np.sqrt(
        np.mean(reference_profile**2)
    )

    relative_error = (
        difference / reference_size * 100
    )

    print(
        f"{sample_number:6d}    "
        f"{relative_error:8.3f}%"
    )


# ==================================================
# 6. 把4条剖面画在同一张图里
# ==================================================

fig, ax = plt.subplots(figsize=(8, 6))

for sample_number in sample_counts:

    ax.plot(
        distance,
        profiles[sample_number],
        marker="o",
        markersize=3,
        label=f"{sample_number} frames"
    )
plt.gcf().patch.set_edgecolor("none")
plt.gcf().patch.set_linewidth(0)

ax.set_xlabel("Y position (pixel)")
ax.set_ylabel("Mean velocity magnitude")
ax.set_title(
    f"Statistical convergence at column {column_number}"
)

ax.grid(True)
ax.legend()

plt.tight_layout()

output_path = folder_path / "profile_convergence.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()