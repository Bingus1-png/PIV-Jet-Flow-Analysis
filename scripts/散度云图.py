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
# 2. 读取数据
# ==================================================

data = sio.loadmat(file_path)


# ==================================================
# 3. 选择需要绘制的帧
# ==================================================

frame_number = 2300
frame_index = frame_number - 1


# ==================================================
# 4. 取出这一帧的 x、y、u、v
# ==================================================

X = np.asarray(data["x"][frame_index, 0], dtype=float)
Y = np.asarray(data["y"][frame_index, 0], dtype=float)

U = np.asarray(data["u"][frame_index, 0], dtype=float)
V = np.asarray(data["v"][frame_index, 0], dtype=float)


# ==================================================
# 5. 取出横纵坐标
# ==================================================

x_coord = X[0, :]
y_coord = Y[:, 0]


# ==================================================
# 6. 计算散度
#
# 散度 = du/dx + dv/dy
# ==================================================

dU_dx = np.gradient(
    U,
    x_coord,
    axis=1,
    edge_order=2
)

dV_dy = np.gradient(
    V,
    y_coord,
    axis=0,
    edge_order=2
)

divergence = dU_dx + dV_dy


# ==================================================
# 7. 设置颜色范围
#
# 使用散度绝对值的第 99 百分位数
# 避免少量异常大值影响整张图的颜色
# ==================================================

div_limit = np.nanpercentile(
    np.abs(divergence),
    99
)

levels = np.linspace(
    -div_limit,
    div_limit,
    61
)


# ==================================================
# 8. 绘制散度云图
# ==================================================

fig, ax = plt.subplots(figsize=(11, 5))

contour = ax.contourf(
    X,
    Y,
    divergence,
    levels=levels,
    cmap="coolwarm",
    extend="both"
)


# ==================================================
# 9. 设置坐标轴
# ==================================================


ax.set_aspect("equal")

ax.set_title(
    f"Frame {frame_number} Instantaneous Divergence"
)

ax.set_xlabel("x (pixel)")
ax.set_ylabel("y (pixel)")


# ==================================================
# 10. 添加颜色条
# ==================================================

colorbar = fig.colorbar(
    contour,
    ax=ax,
    pad=0.02,
    shrink=0.76,
    extendrect=True
)

colorbar.set_label(
    "Divergence (velocity unit / pixel)"
)


# ==================================================
# 11. 保存并显示图片
# ==================================================

fig.tight_layout()

output_path = (
    folder_path
    / f"frame_{frame_number}_divergence.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()