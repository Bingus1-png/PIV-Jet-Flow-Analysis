from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# ==================================================
# 1. 找到 MAT 文件
# ==================================================

# 当前 Python 文件所在的文件夹
folder_path = Path(__file__).resolve().parent

# MAT 文件的完整路径
file_path = folder_path / "PIV3000-clean.mat"


# ==================================================
# 2. 设置需要绘制的帧
# ==================================================

frame_number = 150

# Python数组从0开始，所以要减1
frame_index = frame_number - 1


# ==================================================
# 3. 读取 MAT 文件
# ==================================================

data = sio.loadmat(file_path)


# ==================================================
# 4. 取出第1500帧的 x、y、u、v
# ==================================================

X = np.asarray(data["x"][frame_index, 0], dtype=float)
Y = np.asarray(data["y"][frame_index, 0], dtype=float)

U = np.asarray(data["u"][frame_index, 0], dtype=float)
V = np.asarray(data["v"][frame_index, 0], dtype=float)


# ==================================================
# 5. 取出 x、y 方向坐标
# ==================================================

# X每一行的x坐标相同，所以取第一行
x_coord = X[0, :]

# Y每一列的y坐标相同，所以取第一列
y_coord = Y[:, 0]


# ==================================================
# 6. 计算涡量
#
# omega_z = dV/dx - dU/dy
# ==================================================

# V沿x方向的变化率
dV_dx = np.gradient(
    V,
    x_coord,
    axis=1,
    edge_order=2
)

# U沿y方向的变化率
dU_dy = np.gradient(
    U,
    y_coord,
    axis=0,
    edge_order=2
)

# 二维涡量
omega_z = dV_dx - dU_dy


# ==================================================
# 7. 设置颜色显示范围
# ==================================================

# 取涡量绝对值的99%分位数
# 避免极少数特别大的数值影响整张图的颜色
omega_limit = np.nanpercentile( np.abs(omega_z),99 )

# 让正负涡量使用相同范围
levels = np.linspace(
    -omega_limit,
    omega_limit,
    61
)


# ==================================================
# 8. 绘制涡量云图
# ==================================================

fig, ax = plt.subplots(figsize=(11, 5))

contour = ax.contourf(
    X,
    Y,
    omega_z,
    levels=levels,
    cmap="coolwarm",
    extend="both",
)

# 因为图像坐标中的y通常向下增大


# 保持横纵坐标比例一致
ax.set_aspect("equal")

ax.set_title(
    f"Frame {frame_number} Instantaneous Vorticity"
)

ax.set_xlabel("x (pixel)")
ax.set_ylabel("y (pixel)")


# ==================================================
# 9. 添加颜色条
# ==================================================

colorbar = fig.colorbar(
    contour,
    ax=ax,
    pad=0.02,
    extendrect=True,
    shrink=0.77

)

colorbar.set_label(
    "Vorticity (velocity unit / pixel)"
)


# ==================================================
# 10. 调整排版并保存图片
# ==================================================

fig.tight_layout()

output_path = (
    folder_path
    / f"frame_{frame_number}_vorticity.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("涡量图已经保存到：")
print(output_path)