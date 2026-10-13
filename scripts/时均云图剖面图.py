from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


# ==================================================
# 1. MAT文件位置
# ==================================================

# 当前Python文件所在的文件夹
folder_path = Path(__file__).resolve().parent

# MAT文件的完整路径
file_path = folder_path / "PIV3000-clean.mat"


# ==================================================
# 2. 基本参数
# ==================================================

# 要绘制的剖面列
frame_number = 200
# 日常所说的“第20列”，这里直接填写20
selected_column = 100

# 中值滤波窗口大小：3×3
median_size = 3

# 3σ异常判据
sigma_factor = 3.0


# ==================================================
# 3. 读取MAT文件
# ==================================================

data = sio.loadmat(file_path)

# u和v中包含的总帧数
total_frames = data["u"].shape[0]

print("总帧数：", total_frames)


# ==================================================
# 4. 读取坐标
#
# 每一帧的x、y坐标通常相同，
# 所以这里只读取第一帧
# ==================================================

x = np.asarray(
    data["x"][0, 0],
    dtype=float
).squeeze()

y = np.asarray(
    data["y"][0, 0],
    dtype=float
).squeeze()

number_of_rows, number_of_columns = x.shape

print("单帧数据形状：", x.shape)


# ==================================================
# 5. 检查选择的列是否正确
# ==================================================

if selected_column < 1 or selected_column > number_of_columns:
    raise ValueError(
        f"selected_column必须在1～{number_of_columns}之间，"
        f"当前输入的是{selected_column}"
    )

# Python从0开始编号
# 因此第20列对应索引19
column_index = selected_column - 1


# ==================================================
# 6. 创建累计数组
#
# 分别累计修正前、修正后的u和v，
# 最后除以有效数据数量，得到时均速度分量
# ==================================================

u_before_sum = np.zeros_like(x, dtype=float)
v_before_sum = np.zeros_like(x, dtype=float)

u_after_sum = np.zeros_like(x, dtype=float)
v_after_sum = np.zeros_like(x, dtype=float)

u_before_count = np.zeros_like(x, dtype=int)
v_before_count = np.zeros_like(x, dtype=int)

u_after_count = np.zeros_like(x, dtype=int)
v_after_count = np.zeros_like(x, dtype=int)


# ==================================================
# 7. 逐帧进行异常值判断、修正和累计
# ==================================================

for k in range(total_frames):

    # ----------------------------------------------
    # 读取当前帧原始速度
    # ----------------------------------------------

    u_original = np.asarray(
        data["u"][k, 0],
        dtype=float
    ).squeeze()

    v_original = np.asarray(
        data["v"][k, 0],
        dtype=float
    ).squeeze()

    # 检查形状是否正确
    if u_original.shape != x.shape:
        raise ValueError(
            f"第{k + 1}帧u的形状为{u_original.shape}，"
            f"与坐标形状{x.shape}不一致"
        )

    if v_original.shape != x.shape:
        raise ValueError(
            f"第{k + 1}帧v的形状为{v_original.shape}，"
            f"与坐标形状{x.shape}不一致"
        )

    # ----------------------------------------------
    # 计算3×3局部中值
    # ----------------------------------------------

    u_median = median_filter(
        u_original,
        size=median_size,
        mode="nearest"
    )

    v_median = median_filter(
        v_original,
        size=median_size,
        mode="nearest"
    )

    # ----------------------------------------------
    # 计算原始速度与局部中值的偏差
    # ----------------------------------------------

    u_difference = np.abs(
        u_original - u_median
    )

    v_difference = np.abs(
        v_original - v_median
    )

    # ----------------------------------------------
    # 计算平均偏差加3σ阈值
    # ----------------------------------------------

    u_threshold = (
        np.nanmean(u_difference)
        + sigma_factor * np.nanstd(u_difference)
    )

    v_threshold = (
        np.nanmean(v_difference)
        + sigma_factor * np.nanstd(v_difference)
    )

    # ----------------------------------------------
    # 判断异常速度矢量
    #
    # u或v任意一个分量异常，
    # 就认为该位置的速度矢量异常
    # ----------------------------------------------

    abnormal_mask = (
        (u_difference > u_threshold)
        | (v_difference > v_threshold)
    )

    # ----------------------------------------------
    # 使用局部中值修正异常速度
    # ----------------------------------------------

    u_after = u_original.copy()
    v_after = v_original.copy()

    u_after[abnormal_mask] = (
        u_median[abnormal_mask]
    )

    v_after[abnormal_mask] = (
        v_median[abnormal_mask]
    )

    # ----------------------------------------------
    # 累计修正前的u
    # ----------------------------------------------

    valid_u_before = np.isfinite(u_original)

    u_before_sum[valid_u_before] += (
        u_original[valid_u_before]
    )

    u_before_count[valid_u_before] += 1

    # ----------------------------------------------
    # 累计修正前的v
    # ----------------------------------------------

    valid_v_before = np.isfinite(v_original)

    v_before_sum[valid_v_before] += (
        v_original[valid_v_before]
    )

    v_before_count[valid_v_before] += 1

    # ----------------------------------------------
    # 累计修正后的u
    # ----------------------------------------------

    valid_u_after = np.isfinite(u_after)

    u_after_sum[valid_u_after] += (
        u_after[valid_u_after]
    )

    u_after_count[valid_u_after] += 1

    # ----------------------------------------------
    # 累计修正后的v
    # ----------------------------------------------

    valid_v_after = np.isfinite(v_after)

    v_after_sum[valid_v_after] += (
        v_after[valid_v_after]
    )

    v_after_count[valid_v_after] += 1

    # 每处理500帧显示一次进度
    if (
        (k + 1) % 500 == 0
        or k == total_frames - 1
    ):
        print(
            f"已经处理{k + 1}/{total_frames}帧"
        )


# ==================================================
# 8. 计算修正前后的时均速度分量
# ==================================================

mean_u_before = np.divide(
    u_before_sum,
    u_before_count,
    out=np.full_like(u_before_sum, np.nan),
    where=u_before_count > 0
)

mean_v_before = np.divide(
    v_before_sum,
    v_before_count,
    out=np.full_like(v_before_sum, np.nan),
    where=v_before_count > 0
)

mean_u_after = np.divide(
    u_after_sum,
    u_after_count,
    out=np.full_like(u_after_sum, np.nan),
    where=u_after_count > 0
)

mean_v_after = np.divide(
    v_after_sum,
    v_after_count,
    out=np.full_like(v_after_sum, np.nan),
    where=v_after_count > 0
)


# ==================================================
# 9. 计算修正前后的时均速度幅值
#
# 先计算平均u、平均v，
# 再计算平均速度矢量的幅值
# ==================================================

mean_speed_before = np.hypot(
    mean_u_before,
    mean_v_before
)

mean_speed_after = np.hypot(
    mean_u_after,
    mean_v_after
)

# ==================================================
# 提取第20列时均速度剖面
# ==================================================

selected_column = 100
column_index = selected_column - 1

x_position = np.nanmean(x[:, column_index])

y_profile = y[:, column_index]

profile_before = mean_speed_before[:, column_index]
profile_after = mean_speed_after[:, column_index]


# ==================================================
# 删除无效数据
# ==================================================

valid = (
    np.isfinite(y_profile)
    & np.isfinite(profile_before)
    & np.isfinite(profile_after)
)

y_profile = y_profile[valid]
profile_before = profile_before[valid]
profile_after = profile_after[valid]


# ==================================================
# 按照y坐标排序并计算竖直距离
# ==================================================

order = np.argsort(y_profile)

y_profile = y_profile[order]
profile_before = profile_before[order]
profile_after = profile_after[order]

distance = y_profile - y_profile[0]


# ==================================================
# 计算前后差异和评价指标
# ==================================================

# 保留正负号：
# 正值表示修正后速度更大
# 负值表示修正后速度更小
profile_difference = (
    profile_after - profile_before
)

denominator = np.linalg.norm(profile_before)

if denominator == 0:
    relative_l2 = np.nan
else:
    relative_l2 = (
        np.linalg.norm(profile_difference)
        / denominator
        * 100
    )

maximum_absolute_difference = np.max(
    np.abs(profile_difference)
)

print(
    f"第{selected_column}列归一化L2相对差异："
    f"{relative_l2:.6f}%"
)

print(
    "最大绝对变化："
    f"{maximum_absolute_difference:.6f} px/frame"
)


# ==================================================
# 设置前后两张图相同的纵轴范围
# ==================================================

profile_min = min(
    np.min(profile_before),
    np.min(profile_after)
)

profile_max = max(
    np.max(profile_before),
    np.max(profile_after)
)

padding = max(
    0.05 * (profile_max - profile_min),
    0.05
)

common_ylim = (
    profile_min - padding,
    profile_max + padding
)


# ==================================================
# 绘图
# ==================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)
fig.patch.set_edgecolor("none")
fig.patch.set_linewidth(0)

# --------------------------------------------------
# 左图：修正前
# --------------------------------------------------

axes[0].plot(
    distance,
    profile_before,
    linewidth=2
)

axes[0].set_title(
    "Before correction\n"
    f"Column {selected_column}, "
    f"x = {x_position:.0f} px"
)

axes[0].set_xlabel(
    "Distance along vertical line [px]"
)

axes[0].set_ylabel(
    "Mean velocity magnitude [px/frame]"
)

axes[0].set_ylim(common_ylim)
axes[0].grid(True, alpha=0.4)


# --------------------------------------------------
# 中图：修正后
# --------------------------------------------------

axes[1].plot(
    distance,
    profile_after,
    linewidth=2
)

axes[1].set_title(
    "After correction\n"
    f"Column {selected_column}, "
    f"x = {x_position:.0f} px"
)

axes[1].set_xlabel(
    "Distance along vertical line [px]"
)

axes[1].set_ylabel(
    "Mean velocity magnitude [px/frame]"
)

axes[1].set_ylim(common_ylim)
axes[1].grid(True, alpha=0.4)


# --------------------------------------------------
# 右图：前后变化
# --------------------------------------------------

axes[2].plot(
    distance,
    profile_difference,
    linewidth=2
)

# 零线表示前后完全一致
axes[2].axhline(
    0,
    linewidth=1,
    linestyle="--"
)

axes[2].set_title(
    "Change after correction\n"
    f"Relative L2 = {relative_l2:.4f}%\n"
    f"Max change = "
    f"{maximum_absolute_difference:.5f} px/frame"
)

axes[2].set_xlabel(
    "Distance along vertical line [px]"
)

axes[2].set_ylabel(
    "After − Before [px/frame]"
)

axes[2].grid(True, alpha=0.4)


fig.suptitle(
    "Time-averaged velocity profile comparison",
    fontsize=15
)

plt.tight_layout(
    rect=[0, 0, 1, 0.93]
)

plt.show()