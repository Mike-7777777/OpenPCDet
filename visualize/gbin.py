import numpy as np
import os
import scipy.stats as stats

# 读取 .bin 文件
def read_bin_file(bin_file_path):
    point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud

# 计算点云范围，忽略X轴上的负数值
def compute_point_cloud_range(point_cloud):
    valid_points = point_cloud[point_cloud[:, 0] >= 0]
    min_bound = np.min(valid_points[:, :3], axis=0)
    max_bound = np.max(valid_points[:, :3], axis=0)
    return min_bound, max_bound

# 初始化全局最小值和最大值及对应文件
global_min_bound = np.array([np.inf, np.inf, np.inf])
global_max_bound = np.array([-np.inf, -np.inf, -np.inf])
global_sum = np.zeros(3)
global_count = 0
min_bound_file = ""
max_bound_file = ""

# 用于存储所有文件的最小值和最大值
all_min_bounds = []
all_max_bounds = []
all_ranges = []

# 遍历文件夹中的所有文件
for i in range(2160):
    bin_file_path = f"./velodyne/{i:06d}.bin"
    if os.path.exists(bin_file_path):
        # 读取点云数据
        point_cloud = read_bin_file(bin_file_path)
        
        # 更新全局和计数
        global_sum += np.sum(point_cloud[:, :3], axis=0)
        global_count += point_cloud.shape[0]
        
        # 计算每个文件的最小值和最大值
        min_bound, max_bound = compute_point_cloud_range(point_cloud)
        
        # 更新全局最小值及对应文件
        if np.any(min_bound < global_min_bound):
            global_min_bound = np.minimum(global_min_bound, min_bound)
            min_bound_file = bin_file_path
            
        # 更新全局最大值及对应文件
        if np.any(max_bound > global_max_bound):
            global_max_bound = np.maximum(global_max_bound, max_bound)
            max_bound_file = bin_file_path
        
        # 存储每个文件的最小值和最大值
        all_min_bounds.append(min_bound)
        all_max_bounds.append(max_bound)
        
        # 计算并存储每个文件的范围
        file_range = max_bound - min_bound
        all_ranges.append(file_range)

# 转换为numpy数组以便于计算
all_min_bounds = np.array(all_min_bounds)
all_max_bounds = np.array(all_max_bounds)
all_ranges = np.array(all_ranges)

# 计算全局平均值
global_mean = global_sum / global_count

# 计算所有文件最小值和最大值的均值和标准差
mean_min_bound = np.mean(all_min_bounds, axis=0)
std_min_bound = np.std(all_min_bounds, axis=0)
mean_max_bound = np.mean(all_max_bounds, axis=0)
std_max_bound = np.std(all_max_bounds, axis=0)

# 计算90%、95%和99%的置信区间
confidence_interval_min_bound_90 = stats.norm.interval(0.90, loc=mean_min_bound, scale=std_min_bound)
confidence_interval_max_bound_90 = stats.norm.interval(0.90, loc=mean_max_bound, scale=std_max_bound)
confidence_interval_min_bound_95 = stats.norm.interval(0.95, loc=mean_min_bound, scale=std_min_bound)
confidence_interval_max_bound_95 = stats.norm.interval(0.95, loc=mean_max_bound, scale=std_max_bound)
confidence_interval_min_bound_99 = stats.norm.interval(0.99, loc=mean_min_bound, scale=std_min_bound)
confidence_interval_max_bound_99 = stats.norm.interval(0.99, loc=mean_max_bound, scale=std_max_bound)

# 定义异常文件判定标准（范围超过均值±3个标准差）
mean_range = np.mean(all_ranges, axis=0)
std_range = np.std(all_ranges, axis=0)
outlier_files = []
for i in range(2160):
    bin_file_path = f"./velodyne/{i:06d}.bin"
    if os.path.exists(bin_file_path):
        file_range = all_ranges[i]
        if np.any(file_range > mean_range + 3 * std_range) or np.any(file_range < mean_range - 3 * std_range):
            outlier_files.append(i)

# 合并连续的异常文件序列
merged_outliers = []
if outlier_files:
    start = outlier_files[0]
    end = outlier_files[0]

    for i in range(1, len(outlier_files)):
        if outlier_files[i] == end + 1:
            end = outlier_files[i]
        else:
            merged_outliers.append((start, end))
            start = outlier_files[i]
            end = outlier_files[i]
    merged_outliers.append((start, end))

# 打印90%范围的POINT_CLOUD_RANGE，最多保留两位小数
print(f"POINT_CLOUD_RANGE: [{round(confidence_interval_min_bound_90[0][0], 2)}, {round(confidence_interval_min_bound_90[0][1], 2)}, {round(confidence_interval_min_bound_90[0][2], 2)}, {round(confidence_interval_max_bound_90[1][0], 2)}, {round(confidence_interval_max_bound_90[1][1], 2)}, {round(confidence_interval_max_bound_90[1][2], 2)}]")

# 打印结果
print(f"Global Min bound: {np.round(global_min_bound, 2)} from file {min_bound_file}")
print(f"Global Max bound: {np.round(global_max_bound, 2)} from file {max_bound_file}")
print(f"Global Mean bound: {np.round(global_mean, 2)}")

# 打印全局范围
range_x = round(global_max_bound[0] - global_min_bound[0], 2)
range_y = round(global_max_bound[1] - global_min_bound[1], 2)
range_z = round(global_max_bound[2] - global_min_bound[2], 2)
print(f"Global point cloud range along x-axis: {range_x}")
print(f"Global point cloud range along y-axis: {range_y}")
print(f"Global point cloud range along z-axis: {range_z}")

# 打印平均最小值和最大值
print(f"Mean Min bound: {np.round(mean_min_bound, 2)}")
print(f"Mean Max bound: {np.round(mean_max_bound, 2)}")

# 打印90%、95%和99%置信区间，最多保留两位小数
print(f"90% confidence interval for Min bound: {np.round(confidence_interval_min_bound_90, 2)}")
print(f"90% confidence interval for Max bound: {np.round(confidence_interval_max_bound_90, 2)}")
print(f"95% confidence interval for Min bound: {np.round(confidence_interval_min_bound_95, 2)}")
print(f"95% confidence interval for Max bound: {np.round(confidence_interval_max_bound_95, 2)}")
print(f"99% confidence interval for Min bound: {np.round(confidence_interval_min_bound_99, 2)}")
print(f"99% confidence interval for Max bound: {np.round(confidence_interval_max_bound_99, 2)}")

# 打印合并后的异常文件信息
print(f"Number of outlier files: {len(outlier_files)}")
print("Outlier files:")
for start, end in merged_outliers:
    if start == end:
        print(f"./velodyne/{start:06d}.bin")
    else:
        print(f"./velodyne/{start:06d}.bin ~ ./velodyne/{end:06d}.bin")
