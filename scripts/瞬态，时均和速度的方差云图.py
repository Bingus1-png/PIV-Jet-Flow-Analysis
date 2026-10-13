from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


# ==================================================
# 1. MAT 文件位置
# ==================================================

# 当前Python文件所在的文件夹
folder_path = Path(__file__).resolve().parent

# MAT文件的完整路径
file_path = folder_path / "PIV3000-clean.mat"


# ==================================================
# 2. 基本参数
# ==================================================

# 选择需要展示的帧
frame_number = 200

# 中值滤波窗口大小：3×3
median_size = 3

# 3σ异常判据
sigma_factor = 3.0


# ==================================================
# 3. 读取 MAT 文件
# ==================================================

data = sio.loadmat(file_path)

# 总帧数
total_frames = data["u"].shape[0]

# 检查输入帧数
if frame_number < 1 or frame_number > total_frames:
    raise ValueError(
        f"frame_number 必须在1～{total_frames}之间，"
        f"当前输入的是{frame_number}"
    )

# Python索引从0开始
frame_index = frame_number - 1


# ==================================================
# 4. 读取坐标
#
# x、y每一帧通常相同，所以只读取第一帧
# ==================================================

x = np.asarray(
    data["x"][0, 0],
    dtype=np.float32
)

y = np.asarray(
    data["y"][0, 0],
    dtype=np.float32
)


# ==================================================
# 5. 读取全部速度数据
#
# 最终数组形状：
# (2999, 53, 123)
# ==================================================

u_before_all = np.stack(
    [
        np.asarray(
            data["u"][i, 0],
            dtype=np.float32
        )
        for i in range(total_frames)
    ],
    axis=0
)

v_before_all = np.stack(
    [
        np.asarray(
            data["v"][i, 0],
            dtype=np.float32
        )
        for i in range(total_frames)
    ],
    axis=0
)

print("u数组形状：", u_before_all.shape)
print("v数组形状：", v_before_all.shape)


# ==================================================
# 6. 创建异常值修正后的数组
#
# 必须使用.copy()
# 否则修改after时，before也会跟着改变
# ==================================================

u_after_all = u_before_all.copy()
v_after_all = v_before_all.copy()

# 保存每一帧的异常点数量
abnormal_counts = np.zeros(
    total_frames,
    dtype=int
)


# ==================================================
# 7. 对每一帧进行异常值处理
#
# 处理步骤：
# 1. 计算3×3局部中值
# 2. 计算原始值与局部中值的偏差
# 3. 使用平均偏差 + 3σ作为阈值
# 4. 异常点使用局部中值替换
# ==================================================

for k in range(total_frames):

    # 当前帧原始速度分量
    u_original = u_before_all[k]
    v_original = v_before_all[k]

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
    # 计算原始值与局部中值的绝对偏差
    # ----------------------------------------------

    u_difference = np.abs(
        u_original - u_median
    )

    v_difference = np.abs(
        v_original - v_median
    )

    # ----------------------------------------------
    # 计算3σ阈值
    # ----------------------------------------------

    u_threshold = (
        np.nanmean(u_difference)
        + sigma_factor
        * np.nanstd(u_difference)
    )

    v_threshold = (
        np.nanmean(v_difference)
        + sigma_factor
        * np.nanstd(v_difference)
    )

    # ----------------------------------------------
    # 判断异常点
    #
    # u或v任意一个分量异常，
    # 就认为整个速度矢量异常
    # ----------------------------------------------

    abnormal_mask = (
        (u_difference > u_threshold)
        | (v_difference > v_threshold)
    )

    # ----------------------------------------------
    # 使用局部中值替换异常速度
    # ----------------------------------------------

    u_after_all[k][abnormal_mask] = (
        u_median[abnormal_mask]
    )

    v_after_all[k][abnormal_mask] = (
        v_median[abnormal_mask]
    )

    # 保存当前帧异常点数量
    abnormal_counts[k] = np.count_nonzero(
        abnormal_mask
    )


print()
print("异常值处理完成")
print(
    "平均每帧异常点数量：",
    np.mean(abnormal_counts)
)
print(
    "异常点最多的一帧：",
    np.max(abnormal_counts)
)


# ==================================================
# 8. 计算指定帧的异常数据统计
# ==================================================

# 第200帧异常点数量
frame_abnormal_count = abnormal_counts[
    frame_index
]

# 第200帧总测量点数量
frame_total_count = u_before_all[
    frame_index
].size

# 异常点比例
frame_abnormal_percent = (
    frame_abnormal_count
    / frame_total_count
    * 100
)

# VDP：有效数据比例
frame_vdp = (
    frame_total_count
    - frame_abnormal_count
) / frame_total_count * 100


print()
print(
    f"第{frame_number}帧总测量点数："
    f"{frame_total_count}"
)

print(
    f"第{frame_number}帧异常点数量："
    f"{frame_abnormal_count}"
)

print(
    f"第{frame_number}帧异常点比例："
    f"{frame_abnormal_percent:.2f}%"
)

print(
    f"第{frame_number}帧VDP："
    f"{frame_vdp:.2f}%"
)


# ==================================================
# 9. 计算指定帧的瞬态速度幅值
#
# speed = sqrt(u² + v²)
# ==================================================

instant_speed_before = np.hypot(
    u_before_all[frame_index],
    v_before_all[frame_index]
)

instant_speed_after = np.hypot(
    u_after_all[frame_index],
    v_after_all[frame_index]
)

# 修正前后速度幅值的绝对差
instant_speed_difference = np.abs(
    instant_speed_after
    - instant_speed_before
)


# ==================================================
# 10. 计算时均速度
#
# 先分别对u和v沿时间方向求平均，
# 再计算平均速度矢量的幅值
# ==================================================

mean_u_before = np.nanmean(
    u_before_all,
    axis=0
)

mean_v_before = np.nanmean(
    v_before_all,
    axis=0
)

mean_u_after = np.nanmean(
    u_after_all,
    axis=0
)

mean_v_after = np.nanmean(
    v_after_all,
    axis=0
)

mean_speed_before = np.hypot(
    mean_u_before,
    mean_v_before
)

mean_speed_after = np.hypot(
    mean_u_after,
    mean_v_after
)

mean_speed_difference = np.abs(
    mean_speed_after
    - mean_speed_before
)


# ==================================================
# 11. 计算速度方差
#
# 速度方差 = Var(u) + Var(v)
# ==================================================

variance_before = (
    np.nanvar(
        u_before_all,
        axis=0
    )
    + np.nanvar(
        v_before_all,
        axis=0
    )
)

variance_after = (
    np.nanvar(
        u_after_all,
        axis=0
    )
    + np.nanvar(
        v_after_all,
        axis=0
    )
)

variance_difference = np.abs(
    variance_after
    - variance_before
)


# ==================================================
# 12. 检查修正前后数据是否发生变化
# ==================================================

print()
print(
    "处理前后u是否完全相同：",
    np.array_equal(
        u_before_all,
        u_after_all
    )
)

print(
    "处理前后v是否完全相同：",
    np.array_equal(
        v_before_all,
        v_after_all
    )
)

print(
    "u发生变化的数据数量：",
    np.count_nonzero(
        u_after_all
        - u_before_all
    )
)

print(
    "v发生变化的数据数量：",
    np.count_nonzero(
        v_after_all
        - v_before_all
    )
)


# ==================================================
# 13. 打印统计量前后变化
# ==================================================

print()
print(
    "时均速度最大变化：",
    np.nanmax(
        mean_speed_difference
    )
)

print(
    "时均速度平均变化：",
    np.nanmean(
        mean_speed_difference
    )
)

print(
    "速度方差最大变化：",
    np.nanmax(
        variance_difference
    )
)

print(
    "速度方差平均变化：",
    np.nanmean(
        variance_difference
    )
)


# ==================================================
# 14. 生成前后对比图的共同颜色等级
#
# 使用百分位数限制极端值，
# 避免少数异常大值导致整张图颜色不明显
# ==================================================

def make_common_levels(
        field_before,
        field_after,
        low_percentile=1,
        high_percentile=99,
        number=60
):

    # 将前后两个二维数组展开并合并
    values = np.concatenate(
        [
            np.asarray(
                field_before,
                dtype=float
            ).ravel(),

            np.asarray(
                field_after,
                dtype=float
            ).ravel()
        ]
    )

    # 只保留有限数值
    values = values[
        np.isfinite(values)
    ]

    # 防止没有有效数据
    if values.size == 0:
        return np.linspace(
            0,
            1,
            number
        )

    # 颜色范围下限
    value_min = np.percentile(
        values,
        low_percentile
    )

    # 颜色范围上限
    value_max = np.percentile(
        values,
        high_percentile
    )

    # 防止最大值和最小值相等
    if np.isclose(
        value_min,
        value_max
    ):

        small_value = max(
            abs(value_min) * 1e-6,
            1e-6
        )

        value_min = (
            value_min
            - small_value
        )

        value_max = (
            value_max
            + small_value
        )

    return np.linspace(
        value_min,
        value_max,
        number
    )


# ==================================================
# 15. 生成差值图的颜色等级
# ==================================================

def make_difference_levels(
        difference_field,
        high_percentile=99,
        number=60
):

    values = np.asarray(
        difference_field,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    # 防止没有有效数据
    if values.size == 0:
        return np.linspace(
            0,
            1,
            number
        )

    value_max = np.percentile(
        values,
        high_percentile
    )

    # 如果前后完全相同，
    # 防止颜色等级全部为0
    if (
        not np.isfinite(value_max)
        or np.isclose(value_max, 0)
    ):
        value_max = 1e-6

    return np.linspace(
        0,
        value_max,
        number
    )


# ==================================================
# 16. 绘制“修正前、修正后、绝对差值”云图
#
# annotation_text：
# 用于在第一张图显示异常点统计数据
# ==================================================

def plot_cloud_comparison(
        field_before,
        field_after,
        field_difference,
        main_title,
        value_label,
        common_percentile=99,
        difference_percentile=99,
        annotation_text=None
):

    # ----------------------------------------------
    # 前后两张图使用相同颜色等级
    # ----------------------------------------------

    common_levels = make_common_levels(
        field_before,
        field_after,
        low_percentile=1,
        high_percentile=common_percentile,
        number=60
    )

    # ----------------------------------------------
    # 差值图单独使用颜色等级
    # ----------------------------------------------

    difference_levels = make_difference_levels(
        field_difference,
        high_percentile=difference_percentile,
        number=60
    )

    # 创建三个子图
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        constrained_layout=True
    )
    fig.patch.set_edgecolor("none")
    fig.patch.set_linewidth(0)
    # ----------------------------------------------
    # 修正前云图
    # ----------------------------------------------

    picture_before = axes[0].contourf(
        x,
        y,
        field_before,
        levels=common_levels,
        cmap="turbo",
        extend="both"
    )

    axes[0].set_title(
        main_title
        + "\nBefore correction"
    )

    # ----------------------------------------------
    # 在第一张图左上角显示异常统计数据
    # ----------------------------------------------

    if annotation_text is not None:

        axes[0].text(
            0.02,
            0.98,
            annotation_text,

            # 使用坐标轴比例定位
            transform=axes[0].transAxes,

            ha="left",
            va="top",
            fontsize=10,

            bbox=dict(
                facecolor="white",
                alpha=0.88,
                edgecolor="black",
                boxstyle="round,pad=0.4"
            )
        )

    # ----------------------------------------------
    # 修正后云图
    # ----------------------------------------------

    picture_after = axes[1].contourf(
        x,
        y,
        field_after,
        levels=common_levels,
        cmap="turbo",
        extend="both"
    )

    axes[1].set_title(
        main_title
        + "\nAfter correction"
    )

    # ----------------------------------------------
    # 修正前后绝对差值图
    # ----------------------------------------------

    picture_difference = axes[2].contourf(
        x,
        y,
        field_difference,
        levels=difference_levels,
        cmap="hot",
        extend="max"
    )

    axes[2].set_title(
        main_title
        + "\nAbsolute difference"
    )

    # ----------------------------------------------
    # 坐标设置
    # ----------------------------------------------

    for ax in axes:

        ax.set_xlabel(
            "x / pixel"
        )

        ax.set_ylabel(
            "y / pixel"
        )

        ax.set_aspect(
            "equal"
        )

        # 图像像素坐标通常向下增加
        ax.invert_yaxis()

    # ----------------------------------------------
    # 前两张图共用一个颜色条
    # ----------------------------------------------

    common_colorbar = fig.colorbar(
        picture_after,
        ax=axes[:2],
        shrink=0.35,
        pad=0.03,
        extendrect = True,
    )

    common_colorbar.set_label(
        value_label
    )

    # ----------------------------------------------
    # 差值图使用单独颜色条
    # ----------------------------------------------

    difference_colorbar = fig.colorbar(
        picture_difference,
        ax=axes[2],
        shrink=0.4,
        pad=0.03
    )

    difference_colorbar.set_label(
        "Absolute difference"
    )

    plt.show()


# ==================================================
# 17. 绘制第200帧瞬态速度云图
#
# 这张图显示异常点统计数据
# ==================================================


plot_cloud_comparison(
    field_before=instant_speed_before,
    field_after=instant_speed_after,
    field_difference=instant_speed_difference,

    main_title=(
        "Instantaneous velocity magnitude "
        f"(Frame {frame_number})"
    ),

    value_label="Velocity magnitude",

    common_percentile=99,
    difference_percentile=99,


)


# ==================================================
# 18. 绘制时均速度云图
#
# 不显示第200帧异常统计框
# ==================================================

plot_cloud_comparison(
    field_before=mean_speed_before,
    field_after=mean_speed_after,
    field_difference=mean_speed_difference,

    main_title=(
        "Time-averaged velocity magnitude"
    ),

    value_label=(
        "Mean velocity magnitude"
    ),

    common_percentile=99,
    difference_percentile=99,

    annotation_text=None
)


# ==================================================
# 19. 绘制速度方差云图
#
# common_percentile设为98，
# 让方差主要区域的颜色变化更加明显
# ==================================================

plot_cloud_comparison(
    field_before=variance_before,
    field_after=variance_after,
    field_difference=variance_difference,

    main_title=(
        "Velocity fluctuation variance"
    ),

    value_label=(
        "Velocity variance"
    ),

    common_percentile=98,
    difference_percentile=98,

    annotation_text=None
)