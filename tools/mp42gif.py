import os
from moviepy.editor import VideoFileClip

def mp4_to_gif(mp4_path):
    """
    将 MP4 文件转换为 GIF 并保存到相同目录
    """
    gif_path = os.path.splitext(mp4_path)[0] + '.gif'
    
    # 使用 moviepy 将 mp4 文件转换为 gif
    clip = VideoFileClip(mp4_path)
    clip.write_gif(gif_path, fps=10)  # 可以调整帧率 fps
    
    print(f"转换成功: {mp4_path} -> {gif_path}")

def traverse_and_convert(folder_path):
    """
    递归遍历文件夹，将所有 mp4 文件转换为 gif
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp4"):
                mp4_path = os.path.join(root, file)
                mp4_to_gif(mp4_path)

# 使用示例
folder_path = "playground/dolphins_bench"  # 替换为目标文件夹路径
traverse_and_convert(folder_path)
