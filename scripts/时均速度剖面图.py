from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

folder_path = Path(__file__).resolve().parent
file_path = folder_path / "PIV3000-clean.mat"

column_numbers = [23, 45, 68, 81, 113]
column_indices = [n - 1 for n in column_numbers]

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

y_coords = Y[:, 0]

fig, ax = plt.subplots(figsize=(8, 6))

for col_num, col_idx in zip(column_numbers, column_indices):
    profile = speed_mean[:, col_idx]
    x_pixel = X[0, col_idx]

    ax.plot(
        y_coords,
        profile,
        linewidth=2,
        label=(f'column {col_num}\n')
    )


    print(f'{col_num}列 (x = {x_pixel:.0f} px)')

    peak_idx = np.nanargmax(profile)
    ax.text(
        profile[peak_idx],
        y_coords[peak_idx],
        f"{x_pixel:.0f}px",
        fontsize=8,
        ha="left",
        va="center"
    )
ax.legend(
    loc="center",
    bbox_to_anchor=(0.5, 0.6),
    fontsize=8
)
ax.set_xlabel("y [pixel]")
ax.set_ylabel("Mean speed [px/frame]")
ax.set_title("Original Mean Speed Profiles at Selected Columns")

ax.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()