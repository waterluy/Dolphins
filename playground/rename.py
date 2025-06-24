import os
import re

def rename_files(directory):
    """
    将目录下所有形如 ('dataX',).png 的文件重命名为 dataX.png
    """
    # 正则表达式匹配模式：('dataX',).png
    pattern = re.compile(r"\('(data\d+)',\)\.png")
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # 提取 dataX 部分
            new_name = f"{match.group(1)}.png"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            # 重命名文件
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    target_dir = "playground/camouflage/test_our"
    if os.path.isdir(target_dir):
        rename_files(target_dir)
        print("重命名完成！")
    else:
        print("错误：目录不存在！")