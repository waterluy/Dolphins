import os

def rename_folders(directory):
    # 遍历目录下的所有文件和文件夹
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            new_name = dir_name
            
            # 替换"_exr_"为"_m3_"
            if "_exr_" in new_name:
                new_name = new_name.replace("_exr_", "_m3_")
            
            # 替换"_exrwoori_"为"_m4_"
            if "_exrwoori_" in new_name:
                new_name = new_name.replace("_exrwoori_", "_m4_")
            
            # 替换"_exrwoori1_"为"_m2_"
            if "_exrwoori1_" in new_name:
                new_name = new_name.replace("_exrwoori1_", "_m2_")
            
            # 如果文件夹名称有改变，进行重命名
            if new_name != dir_name:
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    # 在这里指定你要遍历的根目录
    directory = "results"
    rename_folders(directory)
