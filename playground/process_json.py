import json
import re

def natural_sort_key(key):
    """
    自然排序键生成函数（处理数字部分）
    例如 "data2" -> ["data", 2]
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', key)]

def sort_json(data):
    """
    递归排序 JSON 字典的键（自然排序）
    """
    if isinstance(data, dict):
        return {k: sort_json(data[k]) 
                for k in sorted(data.keys(), key=natural_sort_key)}
    elif isinstance(data, list):
        return [sort_json(item) for item in data]
    else:
        return data

def process_json(input_file, output_file, indent=4):
    """
    读取 JSON 文件并重新保存为带缩进的格式
    
    :param input_file: 输入的 JSON 文件路径
    :param output_file: 输出的 JSON 文件路径
    :param indent: 缩进空格数（默认 4）
    """
    try:
        # 1. 读取 JSON 文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 解析 JSON 数据
        
        # 2. （可选）在这里对 data 进行修改
        # 例如：data["new_key"] = "new_value"
        # 2. 按键名递归排序字典
        sorted_data = sort_json(data)
        
        # 3. 重新保存为带缩进的 JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=indent)
        
        print(f"JSON 文件已处理并保存到: {output_file}")
    
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 不存在！")
    except json.JSONDecodeError:
        print(f"错误：文件 {input_file} 不是有效的 JSON 格式！")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    path = "playground/camouflage/clean_perception.json"
    
    # 默认缩进 4 空格（可调整）
    process_json(path, path, indent=4)