import pickle
from pprint import pprint

def load_pkl_file(file_path):
    """加载PKL文件并返回数据"""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def print_data_examples(data, file_name, num_examples=1):
    """打印数据的示例信息"""
    print(f"Examples from {file_name}:")
    if isinstance(data, dict):
        for key, value in list(data.items())[:num_examples]:  # 只打印前几个项
            print(f"{key}:")
            if isinstance(value, list) and len(value) > 0:
                pprint(value[:1], indent=4)  # 只打印列表中的第一个元素
            else:
                pprint(value, indent=4)
            print()
    elif isinstance(data, list):
        for index, item in enumerate(data[:num_examples]):
            print(f"Item {index}:")
            if isinstance(item, dict):
                pprint({k: item[k] for k in list(item.keys())[:3]}, indent=4)  # 只打印字典中的前三个键值对
            else:
                pprint(item, indent=4)
            print()
    else:
        print(f"Data type: {type(data)}")
    print()

def main(file_paths):
    for file_path in file_paths:
        data = load_pkl_file(file_path)
        if data is not None:
            print_data_examples(data, file_path)
        else:
            print(f"Failed to load data from {file_path}")

if __name__ == "__main__":
    pkl_file_paths = [
        'custom_dbinfos_train.pkl',
        'custom_infos_train.pkl',
        'custom_infos_val.pkl'
    ]
    main(pkl_file_paths)
