import os
import csv

# 输出文件名
output_file = 'merged_scores.csv'

# 存储最终的结果
merged_data = []
header_written = False
header = ['num']  # 第一列为序号（即xxx）
top_folder = 'wlu_outputs/250616'

# 遍历当前目录下的所有文件夹
for folder in os.listdir(top_folder):
    cur_folder = os.path.join(top_folder, folder)
    if os.path.isdir(cur_folder) and '-' in folder:
        try:
            attack_name = folder.split('-')[0]
            defense_name = folder.split('-')[-1]
            csv_path = os.path.join(cur_folder, 'bench_score.csv')
            
            if not os.path.exists(csv_path):
                print(f"⚠️ 跳过：{csv_path} 不存在")
                continue

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) < 2:
                    print(f"⚠️ 跳过：{csv_path} 内容不足两行")
                    continue

                # 提取标题和数据行
                if not header_written:
                    header += rows[0]  # 只写一次表头
                    header_written = True

                values = [attack_name + '-' + defense_name] + rows[1]
                merged_data.append(values)

        except Exception as e:
            print(f"❌ 处理 {cur_folder} 时出错: {e}")
# ✅ 按照第一列（num列）排序
merged_data.sort(key=lambda x: x[0])
# 写入最终合并的CSV
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(merged_data)

print(f"✅ 合并完成，共处理 {len(merged_data)} 个文件夹，结果保存为 {output_file}")
