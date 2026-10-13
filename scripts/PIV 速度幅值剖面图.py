from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# 1. MAT文件位置
file_path = Path(__file__).parent / "PIV3000-clean.mat"


# 2. 选择帧和竖直剖面的列
frame_number = 500 # 第几帧：1～2999
column_number = 110   # 第几列：1～123


# 3. 读取MAT文件
data = sio.loadmat(file_path)


# 4. Python从0开始计数，所以减1
frame_index = frame_number - 1
column_index = column_number - 1


# 5. 取出指定帧的数据
x = data["x"][frame_index, 0]
y = data["y"][frame_index, 0]
u = data["u"][frame_index, 0]
v = data["v"][frame_index, 0]


# 6. 计算速度幅值
magnitude = np.sqrt(u**2 + v**2)


# 7. 取指定列的竖直剖面
y_line = y[:, column_index]
magnitude_line = magnitude[:, column_index]


# 8. 让剖面距离从0开始
distance = np.abs(y_line - y_line[0])


# 9. 画一张图
plt.figure(figsize=(8, 6))
plt.gcf().patch.set_edgecolor("none")
plt.gcf().patch.set_linewidth(0)


plt.plot(distance, magnitude_line)

plt.xlabel("Distance along vertical line [px]")
plt.ylabel("Speed magnitude [px/frame]")

plt.title(
    f"Frame {frame_number}, "
    f"vertical column {column_number}, "
    f"x = {x[0, column_index]:.0f} px"
)

plt.grid(True)
plt.tight_layout()
plt.show()