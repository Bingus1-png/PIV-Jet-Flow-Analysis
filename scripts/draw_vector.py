from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# ==================================================
# 1. 设置文件位置和需要绘制的帧
# ==================================================

# MAT 文件和 Python 文件放在同一个文件夹中
file_path = Path(__file__).resolve().parent / "PIV3000-clean.mat"

# 输入想看的帧数
# 这里按照正常习惯从第1帧开始数
frame_number = 1


# ==================================================
# 2. 读取 MAT 文件
# ==================================================

data = sio.loadmat(file_path)

# 检查文件中是否存在 x、y、u、v
required_variables = ["x", "y", "u", "v"]

for variable_name in required_variables:
    if variable_name not in data:
        raise KeyError(f"MAT文件中没有变量：{variable_name}")


# ==================================================
# 3. 检查输入的帧数
# ==================================================

# 一共有多少帧
number_of_frames = data["u"].size

if frame_number < 1 or frame_number > number_of_frames:
    raise ValueError(
        f"frame_number 应该在 1 到 {number_of_frames} 之间，"
        f"你输入的是 {frame_number}"
    )

# Python 的索引从0开始
# 因此第1帧对应索引0
frame_index = frame_number - 1


# ==================================================
# 4. 取出指定帧的 x、y、u、v
# ==================================================

x = np.asarray(
    data["x"].squeeze()[frame_index],
    dtype=float
).squeeze()

y = np.asarray(
    data["y"].squeeze()[frame_index],
    dtype=float
).squeeze()

u = np.asarray(
    data["u"].squeeze()[frame_index],
    dtype=float
).squeeze()

v = np.asarray(
    data["v"].squeeze()[frame_index],
    dtype=float
).squeeze()


# 检查四个矩阵大小是否相同
if not (x.shape == y.shape == u.shape == v.shape):
    raise ValueError(
        "x、y、u、v的矩阵大小不一致：\n"
        f"x: {x.shape}\n"
        f"y: {y.shape}\n"
        f"u: {u.shape}\n"
        f"v: {v.shape}"
    )

print(f"正在绘制第 {frame_number} 帧")
print(f"矩阵大小：{u.shape}")
print(f"箭头总数：{u.size}")


# ==================================================
# 5. 计算速度大小
# ==================================================

speed = np.sqrt(u**2 + v**2)

# 也可以写成：
# speed = np.hypot(u, v)


# ==================================================
# 6. 绘制完整速度矢量图
# ==================================================

# 创建较大的画布
fig, ax = plt.subplots(
    figsize=(16, 8),
    dpi=150
)
plt.gcf().patch.set_edgecolor("none")
plt.gcf().patch.set_linewidth(0)

q = ax.quiver(
    x,
    y,
    u,
    v,
    speed,                     # 使用速度大小控制箭头颜色

    cmap="turbo",              # 颜色映射

    angles="xy",               # 按x、y坐标中的真实方向画箭头
    scale_units="xy",          # 箭头长度使用x、y坐标单位

    scale=0.2,                 # 越小，箭头越长
    width=0.0010,              # 箭杆粗细

    headwidth=3.0,             # 箭头头部宽度
    headlength=4.0,            # 箭头头部长度
    headaxislength=3.5,        # 箭头头部连接位置

    pivot="middle",            # 测量点位于箭头中间
    minlength=0.1,             # 尽量显示较短的箭头
    alpha=0.9                  # 轻微透明，减轻重叠
)


# ==================================================
# 7. 添加颜色条和坐标信息
# ==================================================

colorbar = fig.colorbar(
    q,
    ax=ax,
    pad=0.02,
    shrink=0.7,
)

colorbar.set_label(
    "Velocity magnitude",
    fontsize=12,
)

ax.set_title(
    f"PIV Velocity Vector Field — Frame {frame_number}",
    fontsize=15
)

ax.set_xlabel(
    "x",
    fontsize=12
)

ax.set_ylabel(
    "y",
    fontsize=12
)

# 保证x、y方向比例一致
ax.set_aspect(
    "equal",
    adjustable="box"
)

# 添加较淡的网格
ax.grid(
    True,
    linestyle="--",
    linewidth=0.4,
    alpha=0.25
)

ax.tick_params(
    labelsize=10
)

plt.tight_layout()


# ==================================================
# 8. 保存图片（不需要保存可以注释掉）
# ==================================================

output_path = (
    Path(__file__).resolve().parent
    / f"velocity_vector_frame_{frame_number}.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"图片已保存到：{output_path}")


# ==================================================
# 9. 显示图片
# ==================================================

plt.show()