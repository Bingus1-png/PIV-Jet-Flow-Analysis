from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


# ==================================================
# 1. 找到 MAT 文件
# ==================================================

# 当前Python文件所在的文件夹
folder_path = Path(__file__).resolve().parent

# MAT文件位置
file_path = folder_path / "PIV3000-clean.mat"


# ==================================================
# 2. 异常值处理参数
# ==================================================

# 3×3局部中值窗口
median_size = 3

# 3sigma判据
sigma_factor = 3.0


# ==================================================
# 3. 读取 MAT 文件
# ==================================================

data = sio.loadmat(file_path)

# 总帧数，应当是2999
total_frames = data["u"].shape[0]

print("总帧数：", total_frames)


# ==================================================
# 4. 读取全部u、v速度数据
#
# 最终形状：
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
# 5. 创建异常值修正后的数组
#
# 必须使用copy()
# 否则修改after时，before也会改变
# ==================================================

u_after_all = u_before_all.copy()
v_after_all = v_before_all.copy()

# 记录每一帧发现了多少个异常点
abnormal_counts = np.zeros(
    total_frames,
    dtype=int
)


# ==================================================
# 6. 对全部2999帧进行异常值修正
#
# 方法：
# 1. 计算3×3局部中值
# 2. 计算原速度与局部中值的偏差
# 3. 偏差超过 平均偏差+3倍标准差 时认为异常
# 4. 用局部中值替换异常速度
# ==================================================

for k in range(total_frames):

    # 当前帧原始速度
    u_original = u_before_all[k]
    v_original = v_before_all[k]

    # 3×3局部中值
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

    # 原始值与局部中值的绝对偏差
    u_difference = np.abs(
        u_original - u_median
    )

    v_difference = np.abs(
        v_original - v_median
    )

    # u、v各自的3sigma阈值
    u_threshold = (
        np.nanmean(u_difference)
        + sigma_factor * np.nanstd(u_difference)
    )

    v_threshold = (
        np.nanmean(v_difference)
        + sigma_factor * np.nanstd(v_difference)
    )

    # u或v任意一个分量异常，
    # 就把整个速度矢量判定为异常
    abnormal_mask = (
        (u_difference > u_threshold)
        | (v_difference > v_threshold)
    )

    # 使用局部中值替换异常点
    u_after_all[k][abnormal_mask] = (
        u_median[abnormal_mask]
    )

    v_after_all[k][abnormal_mask] = (
        v_median[abnormal_mask]
    )

    # 记录当前帧异常点数量
    abnormal_counts[k] = np.count_nonzero(
        abnormal_mask
    )

    # 每处理500帧，打印一次进度
    if (
        (k + 1) % 500 == 0
        or k == total_frames - 1
    ):
        print(
            f"已处理 {k + 1} / {total_frames} 帧"
        )


print()
print("异常值处理完成")
print(
    "平均每帧异常点数量：",
    np.mean(abnormal_counts)
)


# ==================================================
# 7. 使用全部2999帧计算参考时均速度场
#
# 参考场：
# 全部2999帧修正后的u、v平均值
# ==================================================

reference_mean_u = np.nanmean(
    u_after_all,
    axis=0,
    dtype=np.float64
)

reference_mean_v = np.nanmean(
    v_after_all,
    axis=0,
    dtype=np.float64
)


# ==================================================
# 8. 计算参考时均速度场的整体大小
#
# 相当于整个53×123速度场的二范数
# ==================================================

reference_norm = np.sqrt(
    np.nansum(
        reference_mean_u**2
        + reference_mean_v**2
    )
)

if np.isclose(reference_norm, 0):
    raise ValueError(
        "参考时均速度场大小为0，无法计算相对误差"
    )


# ==================================================
# 9. 计算前N帧时均速度场的相对误差
#
# N依次取：
# 1, 2, 3, ..., 2999
#
# 相对误差 =
# ||前N帧时均场 - 2999帧时均场||
# -------------------------------- × 100%
#       ||2999帧时均场||
# ==================================================

relative_error = np.zeros(
    total_frames,
    dtype=np.float64
)

# 用于逐帧累加u和v
sum_u = np.zeros_like(
    reference_mean_u,
    dtype=np.float64
)

sum_v = np.zeros_like(
    reference_mean_v,
    dtype=np.float64
)


for k in range(total_frames):

    # 加入第k+1帧
    sum_u += u_after_all[k]
    sum_v += v_after_all[k]

    # 当前使用的帧数
    frame_count = k + 1

    # 前frame_count帧的时均u、v
    current_mean_u = (
        sum_u / frame_count
    )

    current_mean_v = (
        sum_v / frame_count
    )

    # 当前时均场与2999帧参考场的差
    difference_norm = np.sqrt(
        np.nansum(
            (
                current_mean_u
                - reference_mean_u
            )**2
            +
            (
                current_mean_v
                - reference_mean_v
            )**2
        )
    )

    # 相对误差，转换为百分比
    relative_error[k] = (
        difference_norm
        / reference_norm
        * 100
    )


# ==================================================
# 10. 创建1～2999帧的横坐标
# ==================================================

frame_numbers = np.arange(
    1,
    total_frames + 1
)


# ==================================================
# 11. 绘制全部2999个点
# ==================================================

fig = plt.figure(figsize=(12, 6))
fig.patch.set_edgecolor("none")
fig.patch.set_linewidth(0)

plt.plot(
    frame_numbers,
    relative_error,
    linewidth=0.9,
    label="Relative convergence error"
)

plt.xlabel(
    "Number of frames, N",
    fontsize=12
)

plt.ylabel(
    "Relative error / %",
    fontsize=12
)

plt.title(
    "Relative Error of Time-Averaged Velocity Field",
    fontsize=14
)

# 横坐标显示全部1～2999帧
plt.xlim(
    1,
    total_frames
)

# 纵坐标从0开始
plt.ylim(bottom=0)

# 每250帧显示一个横坐标刻度
plt.xticks(
    np.arange(
        0,
        total_frames + 1,
        250
    )
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

# 保存图片
output_picture = (
    folder_path
    / "relative_error_1_to_2999.png"
)

plt.savefig(
    output_picture,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==================================================
# 12. 保存全部2999个相对误差数据
# ==================================================

output_csv = (
    folder_path
    / "relative_error_1_to_2999.csv"
)

np.savetxt(
    output_csv,

    np.column_stack(
        (
            frame_numbers,
            relative_error
        )
    ),

    delimiter=",",

    header=(
        "frame_number,"
        "relative_error_percent"
    ),

    comments=""
)


# ==================================================
# 13. 输出全部2999帧结果
# ==================================================


for n in range(
    1,
    total_frames + 1
):

    print(
        f"前 {n:4d} 帧，"
        f"相对误差 = "
        f"{relative_error[n - 1]:.6f}%"
    )


print()
print("图片保存位置：", output_picture)
print("数据保存位置：", output_csv)