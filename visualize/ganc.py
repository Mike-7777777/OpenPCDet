import os
import numpy as np

# 定义标签文件目录
label_dir = 'label_2'

# 定义目标类别
class_names = ['CAR', 'PEDESTRIAN', 'BICYCLE']

# 初始化存储尺寸的字典
dimensions = {class_name: [] for class_name in class_names}

# 读取所有标签文件
for filename in os.listdir(label_dir):
    filepath = os.path.join(label_dir, filename)
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            obj_class = data[0].upper()
            if obj_class in dimensions:
                height, width, length = map(float, data[8:11])
                dimensions[obj_class].append([height, width, length])

# 计算每个类别的平均尺寸和标准差
anchor_generator_config = []
for class_name in class_names:
    dims = np.array(dimensions[class_name])
    if len(dims) == 0:
        continue
    mean_dims = np.mean(dims, axis=0)
    std_dims = np.std(dims, axis=0)
    
    anchor_config = {
        'class_name': class_name,
        'anchor_sizes': [mean_dims.tolist()],
        'anchor_rotations': [0, 1.57],
        'anchor_bottom_heights': [-mean_dims[0] / 2],
        'align_center': False,
        'feature_map_stride': 2,
        'matched_threshold': 0.5,
        'unmatched_threshold': 0.35
    }
    anchor_generator_config.append(anchor_config)

# 打印配置
import pprint
pprint.pprint(anchor_generator_config)
