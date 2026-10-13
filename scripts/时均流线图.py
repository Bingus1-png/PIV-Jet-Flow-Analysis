from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

folder_path = Path(__file__).resolve().parent
file_path = folder_path / "PIV3000-clean.mat"

data = sio.loadmat(file_path)

x_cells = data["x"]
y_cells = data["y"]
u_cells = data["u"]
v_cells = data["v"]

total_frames = len(u_cells)

X = np.asarray(x_cells[0, 0], dtype=float)
Y = np.asarray(y_cells[0, 0], dtype=float)

U_sum = np.zeros_like(X, dtype=float)
V_sum = np.zeros_like(Y, dtype=float)

for i in range(total_frames):
    U = np.asarray(u_cells[i, 0], dtype=float)
    V = np.asarray(v_cells[i, 0], dtype=float)

    U_sum = U_sum + U
    V_sum = V_sum + V

U_mean = U_sum / total_frames
V_mean = V_sum / total_frames

speed_mean = np.sqrt(U_mean**2 + V_mean**2)

x_line = X[0, :]
y_line = Y[:, 0]

fig, ax = plt.subplots(figsize=(12, 5))

speed_min = np.nanmin(speed_mean)
speed_max = np.nanmax(speed_mean)

levels = np.linspace(speed_min, speed_max, 60)

cf = ax.contourf(
    X,
    Y,
    speed_mean,
    levels=levels,
    cmap="viridis"
)

cbar = fig.colorbar(cf, ax=ax, pad=0.02,shrink=0.87)
cbar.set_label("Mean speed [px/frame]")

ax.streamplot(
    x_line,
    y_line,
    U_mean,
    V_mean,
    density=2.0,
    color="white",
    linewidth=0.8,
    arrowsize=1.1
)

ax.set_xlabel("x [pixel]")
ax.set_ylabel("y [pixel]")
ax.set_title("Time-averaged Streamline Plot")
ax.set_aspect("equal")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("time_averaged_streamline.png", dpi=300, bbox_inches="tight")
plt.show()