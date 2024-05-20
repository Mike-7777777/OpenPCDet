# get model config

import numpy as np
import math

# 定义计算体素尺寸的函数
def calculate_optimal_alpha(real_range, voxel_size):
    n = math.floor(real_range / (voxel_size * 16))
    alpha = n * voxel_size * 16
    return alpha

# 输入的点云范围
point_cloud_range = np.array([0.0, -180.0, -10.0, 160.0, 140.0, 0.0])

# 计算Z轴范围并设置体素尺寸
z_range = point_cloud_range[5] - point_cloud_range[2]
z_voxel_size = z_range  # 体素尺寸为Z轴范围

# 计算X和Y轴的真实点云范围
x_real_range = point_cloud_range[3] - point_cloud_range[0]
y_real_range = point_cloud_range[4] - point_cloud_range[1]

# 初始体素尺寸为0.64
initial_voxel_size = 0.64

# 计算新的点云范围α
x_alpha = calculate_optimal_alpha(x_real_range, initial_voxel_size)
y_alpha = calculate_optimal_alpha(y_real_range, initial_voxel_size)

# 计算减去的值，并在最大值一侧减去
x_reduce = x_real_range - x_alpha
y_reduce = y_real_range - y_alpha

# 确保减去的值不为负
x_reduce = max(x_reduce, 0)
y_reduce = max(y_reduce, 0)

# 更新点云范围，不对0做减法，只在最大值一侧减去所有需要的值
new_point_cloud_range = np.array([
    point_cloud_range[0],
    point_cloud_range[1],
    point_cloud_range[2],
    point_cloud_range[3] - x_reduce,
    point_cloud_range[4] - y_reduce,
    point_cloud_range[5]
])

# 更新体素配置
voxel_size = [initial_voxel_size, initial_voxel_size, z_voxel_size]

# 打印结果
print(f"POINT_CLOUD_RANGE: {new_point_cloud_range.tolist()}")
print(f"    - NAME: transform_points_to_voxels")
print(f"      VOXEL_SIZE: {voxel_size}")
print(f"      MAX_POINTS_PER_VOXEL: 10")
print(f"      MAX_NUMBER_OF_VOXELS: {{'train': 1200000, 'test': 1200000}}")
