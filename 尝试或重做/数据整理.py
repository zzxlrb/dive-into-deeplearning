import os
import shutil
import torch

def organize_files(directory):
    # 获取目录下所有文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 遍历所有文件
    for filename in files:
        # 提取文件名中的数字部分
        parts = filename.split('_')
        if len(parts) == 2 and parts[0].isdigit():
            num = int(parts[0])
            # 创建目标文件夹（如果不存在）
            target_folder = os.path.join(directory, str(num))
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            # 移动文件到目标文件夹
            shutil.move(os.path.join(directory, filename), target_folder)
            print(f"Moved {filename} to {target_folder}")
        else:
            print(f"Ignored {filename}")


directory = 'C:\\Users\\23225\\PycharmProjects\\动手深度学习\尝试或重做\\food\\validation'
#
# # 调用函数来整理文件
organize_files(directory)
