import yaml
import os

def dump_args(folder, args):
    # 将参数转换为字典
    args_dict = vars(args)
    log_file = os.path.join(folder, 'param.yaml')
    # 将参数保存到 log.yaml 文件中
    with open(log_file, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

    print(f"参数已保存到 {log_file}")