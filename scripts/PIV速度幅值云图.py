from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# 数据文件名
# 因为 .py 文件和 .mat 文件在同一个文件夹，所以只写文件名
file_path = Path("PIV3000-clean.mat")

# 要绘制的帧号，范围是 1 到 2999
frame_number = 1


# 检查文件是否存在
if not file_path.exists():
    raise FileNotFoundError(
        "没有找到 PIV3000-clean.mat。\n"
        "请确认 .mat 文件和 Python 文件在同一个项目文件夹中。"
    )


# 读取 MATLAB 文件
data = loadmat(
    file_path,
    variable_names=["x", "y", "u", "v"],
)





# 取出所有帧的数据
x_all = data["x"]
y_all = data["y"]
u_all = data["u"]
v_all = data["v"]


# 检查帧号是否正确
number_of_frames = len(u_all)

if not 1 <= frame_number <= number_of_frames:
    raise ValueError(
        f"帧号必须在 1 到 {number_of_frames} 之间。"
    )


# Python 从 0 开始编号，因此减 1
frame_index = frame_number - 1


# 取出指定帧
x = x_all[frame_index,0]
y = y_all[frame_index,0]
u = u_all[frame_index,0]
v = v_all[frame_index,0]


# 计算速度幅值
speed = np.sqrt(u**2 + v**2)


# 在控制台显示检查信息
print("文件读取成功！")
print("总帧数：", number_of_frames)
print("当前帧：", frame_number)
print("单帧大小：", speed.shape)
print("最小速度幅值：", np.nanmin(speed))
print("最大速度幅值：", np.nanmax(speed))


# 绘制云图
fig = plt.figure(figsize=(12, 5))
fig.patch.set_edgecolor("none")
fig.patch.set_linewidth(0)

cloud = plt.contourf(
    x,
    y,
    speed,
    levels=50,
    cmap="turbo"
)

plt.colorbar(
    cloud,
    label="Velocity magnitude"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title(
    f"PIV velocity magnitude - frame {frame_number}"
)

# 保证横纵坐标比例相同
plt.axis("equal")

# 如果 y 是图像像素坐标，保留这一行
plt.gca().invert_yaxis()

plt.tight_layout()

# 显示图片
plt.show()