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
# 3. 设置需要放大的区域
# ==================================================

# 人习惯的编号：
# 第20～50行，第40～70列，包含两端

# Python从0开始编号，切片结束位置不包含
row_slice = slice(19, 50)
column_slice = slice(39, 70)


# ==================================================
# 4. 读取MAT文件
# ==================================================

data = sio.loadmat(file_path)

required_variables = ["x", "y", "u", "v"]

for variable_name in required_variables:

    if variable_name not in data:

        raise KeyError(
            f"MAT文件中不存在变量：{variable_name}"
        )


# ==================================================
# 5. 检查帧数
# ==================================================

total_frames = data["u"].size

if frame_number < 1 or frame_number > total_frames:

    raise ValueError(
        f"frame_number必须在1～{total_frames}之间，"
        f"当前输入为{frame_number}"
    )


# ==================================================
# 6. 取出指定帧
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
# 7. 检查矩阵大小
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
# 8. 找出有限数据
# ==================================================

finite_mask = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(u)
    & np.isfinite(v)
)

invalid_mask = ~finite_mask

if np.sum(finite_mask) == 0:

    raise ValueError(
        "当前帧没有可用的有限速度数据"
    )


# ==================================================
# 9. 计算速度大小
# ==================================================

speed = np.sqrt(
    u**2 + v**2
)


# ==================================================
# 10. 使用3sigma原则检测异常
# ==================================================

finite_speed = speed[finite_mask]

speed_mean = np.mean(finite_speed)
speed_std = np.std(finite_speed)

sigma_lower_limit = (
    speed_mean
    - sigma_number * speed_std
)

sigma_upper_limit = (
    speed_mean
    + sigma_number * speed_std
)

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
print(
    "3sigma检测出的异常点数：",
    np.sum(sigma_mask)
)


# ==================================================
# 11. 计算局部中值
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
# 12. 计算原始速度与局部中值的差
# ==================================================

u_difference = np.abs(
    u - u_median
)

v_difference = np.abs(
    v - v_median
)


# ==================================================
# 13. 估计周围数据的正常波动水平
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
# 14. 计算局部异常分数
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
# 15. 合并异常位置
# ==================================================

bad_mask = (
    sigma_mask
    | local_mask
    | invalid_mask
)

good_mask = (
    finite_mask
    & (~bad_mask)
)


# ==================================================
# 16. 修正异常速度
# ==================================================

u_corrected = u.copy()
v_corrected = v.copy()

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
# 17. 创建局部区域掩码
# ==================================================

# 先创建一个与u相同大小的False矩阵
region_mask = np.zeros_like(
    u,
    dtype=bool
)

# 把指定区域设置为True
region_mask[
    row_slice,
    column_slice
] = True


# 局部区域中的正常点
good_region_mask = (
    good_mask
    & region_mask
)

# 局部区域中的有限异常点
bad_region_mask = (
    bad_mask
    & finite_mask
    & region_mask
)


# ==================================================
# 18. 统计局部区域中的异常数量
# ==================================================

region_total_number = int(
    np.sum(region_mask)
)

region_bad_number = int(
    np.sum(
        bad_mask
        & region_mask
    )
)

region_valid_number = (
    region_total_number
    - region_bad_number
)

region_vdp = (
    region_valid_number
    / region_total_number
    * 100
)

print()
print(
    "显示区域：第20～50行，第40～70列"
)
print(
    "区域矩阵大小：",
    x[row_slice, column_slice].shape
)
print(
    "区域速度矢量总数：",
    region_total_number
)
print(
    "区域异常矢量数：",
    region_bad_number
)
print(
    f"区域VDP：{region_vdp:.2f}%"
)


# ==================================================
# 19. 计算局部区域的坐标范围
# ==================================================

x_region = x[
    row_slice,
    column_slice
]

y_region = y[
    row_slice,
    column_slice
]

x_region_min = np.nanmin(x_region)
x_region_max = np.nanmax(x_region)

y_region_min = np.nanmin(y_region)
y_region_max = np.nanmax(y_region)


# 给局部区域四周留出一点空白
x_padding = (
    x_region_max - x_region_min
) * 0.06

y_padding = (
    y_region_max - y_region_min
) * 0.06


# ==================================================
# 20. 创建左右两个子图
# ==================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(16, 7),
    sharex=True,
    sharey=True
)

fig.patch.set_edgecolor("none")
fig.patch.set_linewidth(0)

ax_before = axes[0]
ax_after = axes[1]


# ==================================================
# 21. 左图：局部区域中的正常速度
# ==================================================

ax_before.quiver(
    x[good_region_mask],
    y[good_region_mask],
    u[good_region_mask],
    v[good_region_mask],

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
# 22. 左图：局部区域中的异常速度
# ==================================================

ax_before.quiver(
    x[bad_region_mask],
    y[bad_region_mask],
    u[bad_region_mask],
    v[bad_region_mask],

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
# 23. 左图标题
# ==================================================

ax_before.set_title(
    "Before correction\n"
    "Rows 20-50, Columns 40-70\n"
    f"Bad vectors = {region_bad_number}, "
    f"VDP = {region_vdp:.2f}%",
    fontsize=14
)

ax_before.set_xlabel(
    "X",
    fontsize=12
)

ax_before.set_ylabel(
    "Y",
    fontsize=12
)


# ==================================================
# 24. 右图：局部区域中的修正后速度
# ==================================================

corrected_finite_mask = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(u_corrected)
    & np.isfinite(v_corrected)
)

corrected_region_mask = (
    corrected_finite_mask
    & region_mask
)

ax_after.quiver(
    x[corrected_region_mask],
    y[corrected_region_mask],
    u_corrected[corrected_region_mask],
    v_corrected[corrected_region_mask],

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
    "After correction\n"
    "Rows 20-50, Columns 40-70",
    fontsize=14
)

ax_after.set_xlabel(
    "X",
    fontsize=12
)

ax_after.set_ylabel(
    "Y",
    fontsize=12
)

ax_after.tick_params(
    labelleft=True
)


# ==================================================
# 25. 放大局部区域
# ==================================================

for ax in axes:

    ax.set_xlim(
        x_region_min - x_padding,
        x_region_max + x_padding
    )

    ax.set_ylim(
        y_region_min - y_padding,
        y_region_max + y_padding
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
# 26. 调整子图之间的距离
# ==================================================

fig.subplots_adjust(
    left=0.07,
    right=0.98,
    bottom=0.10,
    top=0.86,
    wspace=0.12
)


# 整张图的大标题
fig.suptitle(
    f"Frame {frame_number} - Enlarged Local Velocity Field",
    fontsize=16
)


# ==================================================
# 27. 保存图片
# ==================================================

output_path = (
    folder_path
    / (
        f"frame_{frame_number:04d}_"
        "rows_20_50_columns_40_70_"
        "before_after.png"
    )
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


# ==================================================
# 28. 显示图片
# ==================================================

plt.show()

print()
print("图片保存位置：")
print(output_path)