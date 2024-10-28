import csv
import os

# 读取单个CSV文件中的数据
def read_experiment_results(filepath):
    results = {}
    with open(filepath, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            results[row['task_name']] = row['weighted_score']
    return results

# 写入数据到最终的汇总表格中
def write_combined_results(output_file, data, fieldnames):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# 主函数，用于处理所有实验结果
def main():
    # 定义实验参数
    eps_values = [0.05, 0.1, 0.2, 0.5]
    steps_values = [3, 5, 10, 20, 50, 100]
    dire = "pos"
    lp = "linf"
    attack_methods = ["pgd", "exr", "exrwoori", "exrwoori1"]
    # attack_methods = ["exr", "pgd"]

    # 输出文件路径
    output_file = 'results/exr1022.csv'

    # 输出表格的表头
    fieldnames = ['eps', 'steps', 'lp', 'dire', 'attack_method', 'avg', 'weather', 'traffic_light', 'timeofday', 'scene', 'open_voc_object', 'detailed_description']

    combined_data = []
    
    try:
        for eps in eps_values:
            for steps in steps_values:
                for method in attack_methods:
                    # 构造文件路径
                    filename = f'results/bench_attack_{method}_white_{lp}_eps{eps}_steps{steps}_{dire}/bench_score.csv'
                    
                    # 检查文件是否存在
                    if os.path.exists(filename):
                        # 读取实验结果
                        experiment_results = read_experiment_results(filename)
                        
                        # 构造新的表格行
                        new_row = {
                            'eps': eps,
                            'steps': steps,
                            'lp': lp,
                            'dire': dire,
                            'attack_method': method,
                            'avg': experiment_results.get('avg', ''),
                            'weather': experiment_results.get('weather', ''),
                            'traffic_light': experiment_results.get('traffic_light', ''),
                            'timeofday': experiment_results.get('timeofday', ''),
                            'scene': experiment_results.get('scene', ''),
                            'open_voc_object': experiment_results.get('open_voc_object', ''),
                            'detailed_description': experiment_results.get('detailed_description', '')
                        }
                        
                        # 将数据加入汇总表中
                        combined_data.append(new_row)
                    else:
                        print(f"File {filename} not found, skipping...")
    finally:  
        # 将结果写入新的CSV表格中
        write_combined_results(output_file, combined_data, fieldnames=fieldnames)
        print(f'All data has been written to {output_file}')

if __name__ == '__main__':
    main()
