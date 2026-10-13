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
# 2. 设置要画第几帧
# ==================================================

frame_number = 1230 # 这里可以改成 1～2999，比如 1、500、1000


# ==================================================
# 3. 读取 MATLAB 数据
# ==================================================

data = sio.loadmat(file_path)

x_cells = data["x"]
y_cells = data["y"]
u_cells = data["u"]
v_cells = data["v"]


# ==================================================
# 4. 取出指定帧的数据
# ==================================================

index = frame_number - 1

X = np.asarray(x_cells[index, 0], dtype=float)
Y = np.asarray(y_cells[index, 0], dtype=float)
U = np.asarray(u_cells[index, 0], dtype=float)
V = np.asarray(v_cells[index, 0], dtype=float)


# ==================================================
# 5. 计算这一帧的速度大小
# ==================================================

speed = np.sqrt(U**2 + V**2)


# ==================================================
# 6. 准备 streamplot 需要的一维坐标
# ==================================================

x_line = X[0, :]
y_line = Y[:, 0]


# ==================================================
# 7. 画瞬时流线图
# ==================================================

fig, ax = plt.subplots(figsize=(12, 5))

speed_min = np.nanmin(speed)
speed_max = np.nanmax(speed)

levels = np.linspace(speed_min, speed_max, 60)

cf = ax.contourf(
    X,
    Y,
    speed,
    levels=levels,
    cmap="viridis",
)

cbar = fig.colorbar(cf, ax=ax, pad=0.02,shrink=0.87
                    )
cbar.set_label("Instantaneous speed [px/frame]")

ax.streamplot(
    x_line,
    y_line,
    U,
    V,
    density=2.0,
    color="white",
    linewidth=0.8,
)

ax.set_xlabel("x [pixel]")
ax.set_ylabel("y [pixel]")
ax.set_title(f"Instantaneous Streamline Plot (Frame {frame_number})")
ax.set_aspect("equal")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f"instant_streamline_frame{frame_number}.png", dpi=300, bbox_inches="tight")
plt.show()