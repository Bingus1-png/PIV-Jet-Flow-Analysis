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

x_cells = data["x"]
y_cells = data["y"]
u_cells = data["u"]
v_cells = data["v"]

total_frames = len(u_cells)

print(f"总帧数：{total_frames}")


# ==================================================
# 3. 取第一帧的坐标
# ==================================================

X = np.asarray(x_cells[0, 0], dtype=float)
Y = np.asarray(y_cells[0, 0], dtype=float)

print("单帧矩阵大小：", X.shape)


# ==================================================
# 4. 计算时均速度场
# ==================================================

U_sum = np.zeros_like(X, dtype=float)
V_sum = np.zeros_like(Y, dtype=float)

for i in range(total_frames):
    U = np.asarray(u_cells[i, 0], dtype=float)
    V = np.asarray(v_cells[i, 0], dtype=float)

    U_sum += U
    V_sum += V

U_mean = U_sum / total_frames
V_mean = V_sum / total_frames

# 速度幅值
speed_mean = np.sqrt(U_mean**2 + V_mean**2)

# ==================================================
# 4.5 自动估算喷口出口直径 D
# ==================================================

# 如果射流从左往右喷，喷口出口通常在最左边
inlet_col_idx = 0

# 喷口出口位置 x0，取最左边这一列的 x 坐标
x0_pixel = X[0, inlet_col_idx]

# y方向坐标
y_coords = Y[:, 0]

# 取喷口出口附近这一列的速度剖面
inlet_profile = speed_mean[:, inlet_col_idx].astype(float)

# 对入口速度剖面做一个简单平滑，减少噪声影响
window = 5
kernel = np.ones(window) / window
inlet_profile_smooth = np.convolve(inlet_profile, kernel, mode="same")

# 用入口最大速度的 50% 作为喷口边界判断标准
threshold_ratio = 0.5
threshold = threshold_ratio * np.nanmax(inlet_profile_smooth)

# 找到速度大于等于 50% 最大速度的位置
above = np.where(inlet_profile_smooth >= threshold)[0]

if len(above) < 2:
    raise ValueError("无法自动估算喷口直径 D，请检查入口速度剖面或手动设置 D")

# 如果有多段区域，选择最长的那一段，避免噪声干扰
split_points = np.where(np.diff(above) > 1)[0] + 1
segments = np.split(above, split_points)
main_segment = max(segments, key=len)

# 这段区域的上下边界
top_idx = main_segment[0]
bottom_idx = main_segment[-1]

y_top = y_coords[top_idx]
y_bottom = y_coords[bottom_idx]

# 喷口直径 D，单位 pixel
D_pixel = abs(y_bottom - y_top)

print("\n自动估算喷口出口结果：")
print(f"喷口出口位置 x0 = {x0_pixel:.2f} px")
print(f"y_top = {y_top:.2f} px")
print(f"y_bottom = {y_bottom:.2f} px")
print(f"喷口直径 D = {D_pixel:.2f} px")
print(f"判断标准：入口速度 >= {threshold_ratio:.1f} × 入口最大速度")

# ==================================================
# 5. 选择要比较的剖面列
# ==================================================

# 这里是第20、40、60、80、100列
# 注意：Python索引从0开始，所以第20列对应索引19
column_numbers = [23, 45, 68, 81, 113]
column_indices = [n - 1 for n in column_numbers]

# y方向坐标
y_coords = Y[:, 0]


# ==================================================
# 6. 画原始时均速度剖面图
# ==================================================

fig, ax = plt.subplots(figsize=(8, 5))

for col_num, col_idx in zip(column_numbers, column_indices):
    profile = speed_mean[:, col_idx]
    x_pixel = X[0, col_idx]

    ax.plot(
        y_coords,
        profile,
        linewidth=2,
        label=f"Column {col_num}, x={x_pixel:.0f}px"
    )

ax.set_xlabel("y [pixel]")
ax.set_ylabel("Mean speed")
ax.set_title("Original Mean Speed Profiles")
ax.grid(True, alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

plt.tight_layout()
plt.savefig("original_mean_speed_profiles_corrected.png", dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# 7. 画归一化后的自相似剖面图
# ==================================================

fig, ax = plt.subplots(figsize=(8, 5))

print("\n检查每条剖面归一化后的最大值：")

for col_num, col_idx in zip(column_numbers, column_indices):

    # 取当前这一列的速度剖面
    profile = speed_mean[:, col_idx].astype(float)

    # ==================================================
    # 7.1 每条剖面自己的最大速度
    # ==================================================

    profile_max = np.nanmax(profile)

    if profile_max == 0 or not np.isfinite(profile_max):
        print(f"Column {col_num}: 最大速度异常，跳过")
        continue

    # 速度归一化
    # 归一化后，这条剖面的最大值应该等于1
    profile_norm = profile / profile_max

    # ==================================================
    # 7.2 找最大速度所在的位置
    # ==================================================

    center_idx = np.nanargmax(profile)


    # ==================================================
    # 7.3 计算半宽度 b
    # ==================================================

    # 找到归一化速度大于等于0.5的区域
    # 也就是速度大于等于最大速度一半的位置
    above_half = np.where(profile_norm >= 0.5)[0]

    if len(above_half) < 2:
        print(f"Column {col_num}: 半宽度无法计算，跳过")
        continue

    # 这个区域在y方向上的两端
    y_min_half = y_coords[above_half[0]]
    y_max_half = y_coords[above_half[-1]]

    y_center = (y_max_half + y_min_half) / 2

    # 半宽度 b = 半高全宽 / 2
    b = abs(y_max_half - y_min_half) / 2

    if b == 0 or not np.isfinite(b):
        print(f"Column {col_num}: b异常，跳过")
        continue

    # ==================================================
    # 7.4 宽度归一化
    # ==================================================

    eta = (y_coords - y_center) / b

    # 打印检查结果
    print(
        f"Column {col_num}: "
        f"profile_max = {profile_max:.4f}, "
        f"max(profile_norm) = {np.nanmax(profile_norm):.4f}, "
        f"y_center = {y_center:.2f}, "
        f"b = {b:.2f}"
    )

    # 画归一化后的剖面
    ax.plot(
        eta,
        profile_norm,
        linewidth=2,
        label=f"Column {col_num}"
    )


# 辅助线
ax.axhline(1.0, linestyle="--", linewidth=1, alpha=0.5)
ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.5)
ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.5)
ax.axvline(-1.0, linestyle="--", linewidth=1, alpha=0.5)
ax.axvline(1.0, linestyle="--", linewidth=1, alpha=0.5)

ax.set_xlabel(r"$\eta = (y-y_c)/b$")
ax.set_ylabel("Mean speed / maximum mean speed")
ax.set_title("Self-similar Normalized Mean Speed Profiles")
ax.set_ylim(0, 1.08)
ax.grid(True, alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

plt.tight_layout()
plt.savefig("self_similarity_profiles_corrected.png", dpi=300, bbox_inches="tight")
plt.show()
