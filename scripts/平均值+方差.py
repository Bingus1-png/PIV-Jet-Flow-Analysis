from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# ==================================================
# 1. 找到 MAT 文件
# ==================================================

folder_path = Path(__file__).resolve().parent
file_path = folder_path / "PIV3000-clean.mat"


# ==================================================
# 2. 读取 MAT 文件
# ==================================================

data = sio.loadmat(file_path)


# ==================================================
# 3. 取出 x、y、u、v
# ==================================================

#x,
x = data["x"][0, 0]
y = data["y"][0, 0]

# 原来的u和v是2999×1的cell
u_cells = data["u"].squeeze()
v_cells = data["v"].squeeze()

total_frames = len(u_cells)

print("总帧数：", total_frames)
print("每一帧的尺寸：", u_cells[0].shape)


# ==================================================
# 4. 把2999个二维矩阵叠成三维数组
# ==================================================

# 原来：
# 第1帧   53×123
# 第2帧   53×123
# ...
# 第2999帧 53×123
#
# 叠起来以后：
# 2999×53×123

u_all = np.stack(u_cells, axis=0)
v_all = np.stack(v_cells, axis=0)

print("u_all的尺寸：", u_all.shape)
print("v_all的尺寸：", v_all.shape)


# ==================================================
# 5. 沿着2999帧的方向计算平均值
# ==================================================

# axis=0表示沿第一个方向，也就是沿2999帧计算
# 结果仍然是53×123

u_mean = np.mean(u_all, axis=0)
v_mean = np.mean(v_all, axis=0)

print("平均u的尺寸：", u_mean.shape)
print("平均v的尺寸：", v_mean.shape)


# ==================================================
# 6. 计算速度方差
# ==================================================

# ddof=1表示计算样本方差
# 结果仍然是53×123

u_variance = np.var(u_all, axis=0, ddof=1)
v_variance = np.var(v_all, axis=0, ddof=1)

print("u方差的尺寸：", u_variance.shape)
print("v方差的尺寸：", v_variance.shape)


# ==================================================
# 7. 计算平均速度矢量的大小
# ==================================================

mean_speed = np.sqrt(u_mean**2 + v_mean**2)


# ==================================================
# 9. 绘制结果
# ==================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10),
    constrained_layout=True
)

# ==================================================
# 8. 设置方差云图的颜色显示范围
# ==================================================

# 使用99%分位数作为颜色条上限
# 避免少数特别大的错误值把颜色范围拉得太大
u_var_max = np.percentile(u_variance, 99)
v_var_max = np.percentile(v_variance, 99)

# 在0到颜色条上限之间，均匀划分50个颜色等级
u_var_levels = np.linspace(0, u_var_max, 50)
v_var_levels = np.linspace(0, v_var_max, 50)

print("u方差颜色条上限：", u_var_max)
print("v方差颜色条上限：", v_var_max)


# ==================================================
# 9. 绘制结果
# ==================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10),
    constrained_layout=True
)
plt.gcf().patch.set_edgecolor("none")
plt.gcf().patch.set_linewidth(0)

# --------------------------------------------------
# 图1：平均速度矢量图
# --------------------------------------------------

speed_plot = axes[0, 0].contourf(
    x,
    y,
    mean_speed,
    levels=50,
    cmap="jet"
)

axes[0, 0].quiver(
    x,
    y,
    u_mean,
    v_mean,
    color="black",
    pivot="mid"
)

fig.colorbar(
    speed_plot,
    shrink=0.5,
    ax=axes[0, 0],
    label="Mean velocity magnitude"
)

axes[0, 0].set_title("Mean velocity field")
axes[0, 0].set_xlabel("x / pixel")
axes[0, 0].set_ylabel("y / pixel")
axes[0, 0].set_aspect("equal")
axes[0, 0].invert_yaxis()


# --------------------------------------------------
# 图2：平均速度大小云图
# --------------------------------------------------

mean_plot = axes[0, 1].contourf(
    x,
    y,
    mean_speed,
    levels=50,
    cmap="jet"
)

fig.colorbar(
    mean_plot,
    shrink=0.5,
    ax=axes[0, 1],
    label="Mean velocity magnitude"
)

axes[0, 1].set_title("Mean velocity magnitude")
axes[0, 1].set_xlabel("x / pixel")
axes[0, 1].set_ylabel("y / pixel")
axes[0, 1].set_aspect("equal")
axes[0, 1].invert_yaxis()


# --------------------------------------------------
# 图3：u方向速度方差
# --------------------------------------------------
vmax_u = np.nanpercentile(u_variance, 99)
levels_u = np.linspace(0, vmax_u, 100)

var_u_plot = np.clip(u_variance, 0, vmax_u)

cf_u = axes[1, 0].contourf(
    x, y,
    var_u_plot,
    levels=levels_u,
    cmap="jet"
)

cbar_u = fig.colorbar(
    cf_u,
    ax=axes[1, 0],
    shrink=0.5,
    label="Variance of u"
)

axes[1, 0].set_title("Variance of u")
axes[1, 0].set_xlabel("x / pixel")
axes[1, 0].set_ylabel("y / pixel")
axes[1, 0].set_aspect("equal")
axes[1, 0].invert_yaxis()


# --------------------------------------------------
# 图4：v方向速度方差
# --------------------------------------------------

vmax_v = np.nanpercentile(v_variance, 99)
levels_v = np.linspace(0, vmax_v, 100)

var_v_plot = np.clip(v_variance, 0, vmax_v)

cf_v = axes[1, 1].contourf(
    x, y,
    var_v_plot,
    levels=levels_v,
    cmap="jet"
)

cbar_v = fig.colorbar(
    cf_v,
    ax=axes[1, 1],
    shrink=0.5,
    label="Variance of v"
)

axes[1, 1].set_title("Variance of v")
axes[1, 1].set_xlabel("x / pixel")
axes[1, 1].set_ylabel("y / pixel")
axes[1, 1].set_aspect("equal")
axes[1, 1].invert_yaxis()


# ==================================================
# 10. 保存和显示图片
# ==================================================
output_path = folder_path / "question1_mean_and_variance.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("图片保存位置：", output_path)