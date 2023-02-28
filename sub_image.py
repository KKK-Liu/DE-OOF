# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.patches import ConnectionPatch
# import  pandas as pd

# MAX_EPISODES = 300
# x_axis_data = []
# for l in range(MAX_EPISODES):
#     x_axis_data.append(l)

# fig, ax = plt.subplots(1, 1)
# # data1 = pd.read_csv('./result/test_reward.csv')['test_reward'].values.tolist()[:MAX_EPISODES]
# # data2 = pd.read_csv('./result/test_reward_att.csv')['test_reward_att'].values.tolist()[:MAX_EPISODES]

# data1 = list(np.random.rand(MAX_EPISODES) * 2 + 3)
# data2 = list(np.random.rand(MAX_EPISODES) * 3)
# ax.plot(data1,label="no att")
# ax.plot(data2,label = "att")
# ax.legend()

# #插入子坐标系
# axins = inset_axes(ax, width="40%", height="20%", loc=3,
#                    bbox_to_anchor=(0.3, 0.1, 2, 2),
#                    bbox_transform=ax.transAxes)
# #在子坐标系中放入数据
# axins.plot(data1)
# axins.plot(data2)


# #设置放大区间
# zone_left = 150
# zone_right = 170
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0.05  # y轴显示范围的扩展比例

# # X轴的显示范围
# xlim0 = x_axis_data[zone_left]-(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio
# xlim1 = x_axis_data[zone_right]+(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio

# # Y轴的显示范围
# y = np.hstack((data1[zone_left:zone_right], data2[zone_left:zone_right]))
# ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
# ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# # 调整子坐标系的显示范围
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)


# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import  pandas as pd

MAX_EPISODES = 300
x_axis_data = []
for l in range(MAX_EPISODES):
    x_axis_data.append(l)

fig, ax = plt.subplots(1, 1)
data1 = list(np.random.rand(MAX_EPISODES) * 2 + 3)
data2 = list(np.random.rand(MAX_EPISODES) * 3)
ax.plot(data1,label="no att")
ax.plot(data2,label = "att")
ax.legend()

#插入子坐标系
axins = inset_axes(ax, width="10%", height="40%", loc=3,
                   bbox_to_anchor=(0.6, 0.3, 5, 2),
                   bbox_transform=ax.transAxes)
#在子坐标系中放入数据
axins.plot(data1)
axins.plot(data2)

#设置放大区间
zone_left = 150
zone_right = 170
# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0  # x轴显示范围的扩展比例
y_ratio = 0.05  # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = x_axis_data[zone_left]-(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio
xlim1 = x_axis_data[zone_right]+(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio

# Y轴的显示范围
y = np.hstack((data1[zone_left:zone_right], data2[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)


# 原图中画方框
tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"blue")

# 画两条线
# 第一条线
xy = (xlim0,ylim0)
xy2 = (xlim0,ylim1)
"""
xy为主图上坐标，xy2为子坐标系上坐标，axins为子坐标系，ax为主坐标系。
"""
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)

axins.add_artist(con)
# 第二条线
xy = (xlim1,ylim0)
xy2 = (xlim1,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

plt.show()