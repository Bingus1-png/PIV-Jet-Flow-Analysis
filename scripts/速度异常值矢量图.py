from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


# ==================================================
# 1. 文件位置
# ==================================================

folder_path = Path(__file__).resolve().parent
file_path = folder_path / "PIV3000-clean.mat"
1

# ==================================================
# 2. 基本参数
# ==================================================

frame_number = 1500

# 中值滤波窗口：3×3
median_size = 3

# 局部中值异常分数阈值
local_threshold = 3.0

# 3sigma中的倍数
sigma_number = 3.0

# 防止除数为0
epsilon = 0.1

# 绘图参数
vector_scale = 0.08
normal_width = 0.0010
bad_width = 0.0025


# ==================================================
# 3. 读取MAT文件
# ==================================================

data = sio.loadmat(file_path)

required_variables = ["x", "y", "u", "v"]

for variable_name in required_variables:
    if variable_name not in data:
        raise KeyError(
            f"MAT文件中不存在变量：{variable_name}"
        )


# ==================================================
# 4. 检查帧数
# ==================================================

total_frames = data["u"].size

if frame_number < 1 or frame_number > total_frames:
    raise ValueError(
        f"frame_number必须在1～{total_frames}之间，"
        f"当前输入为{frame_number}"
    )


# ==================================================
# 5. 取出指定帧
# ==================================================

frame_index = frame_number - 1

x = np.asarray(
    data["x"].flat[frame_index],
    dtype=float
).squeeze()

y = np.asarray(
    data["y"].flat[frame_index],
    dtype=float
).squeeze()

u = np.asarray(
    data["u"].flat[frame_index],
    dtype=float
).squeeze()

v = np.asarray(
    data["v"].flat[frame_index],
    dtype=float
).squeeze()


# ==================================================
# 6. 检查矩阵大小
# ==================================================

if not (x.shape == y.shape == u.shape == v.shape):
    raise ValueError(
        "x、y、u、v矩阵尺寸不一致。\n"
        f"x：{x.shape}\n"
        f"y：{y.shape}\n"
        f"u：{u.shape}\n"
        f"v：{v.shape}"
    )

if u.ndim != 2:
    raise ValueError(
        f"当前速度数据不是二维矩阵，形状为：{u.shape}"
    )


# ==================================================
# 7. 找出有限数据
# ==================================================

finite_mask = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(u)
    & np.isfinite(v)
)

invalid_mask = ~finite_mask

if np.sum(finite_mask) == 0:
    raise ValueError("当前帧没有可用的有限速度数据")


# ==================================================
# 8. 计算速度大小
# ==================================================

speed = np.sqrt(
    u**2 + v**2
)


# ==================================================
# 9. 使用3sigma原则检测异常
# ==================================================

# 只使用正常有限数据计算平均值和标准差
finite_speed = speed[finite_mask]

speed_mean = np.mean(finite_speed)
speed_std = np.std(finite_speed)

# 3sigma上下限
sigma_lower_limit = (
    speed_mean - sigma_number * speed_std
)

sigma_upper_limit = (
    speed_mean + sigma_number * speed_std
)

# 超出平均值±3个标准差的点
sigma_mask = (
    finite_mask
    & (
        (speed < sigma_lower_limit)
        | (speed > sigma_upper_limit)
    )
)

print("速度平均值：", speed_mean)
print("速度标准差：", speed_std)
print("3sigma下限：", sigma_lower_limit)
print("3sigma上限：", sigma_upper_limit)
print("3sigma检测出的异常点数：", np.sum(sigma_mask))


# ==================================================
# 10. 计算局部中值
# ==================================================

u_median = median_filter(
    u,
    size=median_size,
    mode="nearest"
)

v_median = median_filter(
    v,
    size=median_size,
    mode="nearest"
)


# ==================================================
# 11. 计算原始速度与局部中值的差
# ==================================================

u_difference = np.abs(
    u - u_median
)

v_difference = np.abs(
    v - v_median
)


# ==================================================
# 12. 估计周围数据的正常波动水平
# ==================================================

u_difference_median = median_filter(
    u_difference,
    size=median_size,
    mode="nearest"
)

v_difference_median = median_filter(
    v_difference,
    size=median_size,
    mode="nearest"
)


# ==================================================
# 13. 计算局部异常分数
# ==================================================

u_score = (
    u_difference
    / (u_difference_median + epsilon)
)

v_score = (
    v_difference
    / (v_difference_median + epsilon)
)

local_score = np.sqrt(
    u_score**2 + v_score**2
)

local_mask = (
    finite_mask
    & (local_score > local_threshold)
)

print(
    "局部中值方法检测出的异常点数：",
    np.sum(local_mask)
)


# ==================================================
# 14. 合并异常位置
# ==================================================

# | 表示“或者”
#
# 满足下列任何一个条件，就标记为异常：
# 1. 3sigma异常
# 2. 局部中值异常
# 3. NaN或无穷大
bad_mask = (
    sigma_mask
    | local_mask
    | invalid_mask
)

# 能正常绘制的正常点
good_mask = (
    finite_mask
    & (~bad_mask)
)


# ==================================================
# 15. 统计异常数量和VDP
# ==================================================

bad_vector_number = int(
    np.sum(bad_mask)
)

total_vector_number = u.size

valid_vector_number = (
    total_vector_number
    - bad_vector_number
)

vdp = (
    valid_vector_number
    / total_vector_number
    * 100
)

print()
print("当前帧：", frame_number)
print("矩阵大小：", u.shape)
print("速度矢量总数：", total_vector_number)
print("合并后的异常矢量数：", bad_vector_number)
print(f"VDP：{vdp:.2f}%")


# ==================================================
# 16. 修正异常速度
# ==================================================

u_corrected = u.copy()
v_corrected = v.copy()

# 对有限异常值，用局部中值替换
replace_mask = (
    bad_mask
    & finite_mask
)

u_corrected[replace_mask] = (
    u_median[replace_mask]
)

v_corrected[replace_mask] = (
    v_median[replace_mask]
)


# ==================================================
# 17. 创建左右两个子图
# ==================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(16, 6.5),
    sharex=True,
    sharey=True
)


plt.gcf().patch.set_edgecolor("none")
plt.gcf().patch.set_linewidth(0)

ax_before = axes[0]
ax_after = axes[1]


# ==================================================
# 18. 左图：修正前的正常速度
# ==================================================

ax_before.quiver(
    x[good_mask],
    y[good_mask],
    u[good_mask],
    v[good_mask],

    color="black",

    angles="xy",
    scale_units="xy",
    scale=vector_scale,

    pivot="tail",

    width=normal_width,

    headwidth=3,
    headlength=4,
    headaxislength=3.5,

    minlength=0
)


# ==================================================
# 19. 左图：修正前的异常速度
# ==================================================

# 只能绘制有限异常值
bad_finite_mask = (
    bad_mask
    & finite_mask
)

ax_before.quiver(
    x[bad_finite_mask],
    y[bad_finite_mask],
    u[bad_finite_mask],
    v[bad_finite_mask],

    color="red",

    angles="xy",
    scale_units="xy",
    scale=vector_scale,

    pivot="tail",

    width=bad_width,

    headwidth=3.5,
    headlength=4.5,
    headaxislength=4,

    minlength=0,

    zorder=3
)


# ==================================================
# 20. 左图标题
# ==================================================

ax_before.set_title(
    "Before correction\n"
    f"Bad vectors = {bad_vector_number}, "
    f"VDP = {vdp:.2f}%",
    fontsize=14
)

ax_before.set_xlabel("X")
ax_before.set_ylabel("Y")


# ==================================================
# 21. 右图：修正后的速度
# ==================================================

corrected_finite_mask = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(u_corrected)
    & np.isfinite(v_corrected)
)

ax_after.quiver(
    x[corrected_finite_mask],
    y[corrected_finite_mask],
    u_corrected[corrected_finite_mask],
    v_corrected[corrected_finite_mask],

    color="black",

    angles="xy",
    scale_units="xy",
    scale=vector_scale,

    pivot="tail",

    width=normal_width,

    headwidth=3,
    headlength=4,
    headaxislength=3.5,

    minlength=0
)

ax_after.set_title(
    "After correction",
    fontsize=14
)

ax_after.set_xlabel("X")
ax_after.set_ylabel("Y")

ax_after.tick_params(
    labelleft=True
)


# ==================================================
# 22. 设置坐标范围、比例和网格
# ==================================================

for ax in axes:

    ax.set_xlim(
        -70,
        2080
    )

    ax.set_ylim(
        200,
        1820
    )

    ax.set_xticks(
        np.arange(0, 2001, 250)
    )

    ax.set_yticks(
        np.arange(400, 1801, 200)
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.grid(
        True,
        linewidth=0.8,
        alpha=0.55
    )

    ax.tick_params(
        labelsize=10
    )


# ==================================================
# 23. 调整子图距离
# ==================================================

fig.subplots_adjust(
    left=0.06,
    right=0.98,
    bottom=0.10,
    top=0.90,
    wspace=0.10
)


# ==================================================
# 24. 保存图片
# ==================================================

output_path = (
    folder_path
    / f"frame_{frame_number:04d}_3sigma_median_correction.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


# ==================================================
# 25. 显示图片
# ==================================================

plt.show()

print()
print("图片保存位置：")
print(output_path)
