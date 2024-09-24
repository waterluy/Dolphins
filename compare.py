import pandas as pd

# 读取CSV文件
file1 = 'csvfiles/conversation_inference.csv'
file2 = 'csvfiles/conversation_attack.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

min_rows = min(len(df1), len(df2))

# 遍历最少行数的范围内，比较'dolphins answer'列
for index in range(min_rows):
    answer1 = df1['dolphins answer'].iloc[index]
    answer2 = df2['dolphins answer'].iloc[index]
    
    if answer1 != answer2:
        print(f"行号: {index + 1}")
        print(f"文件1的内容: {answer1}")
        print(f"文件2的内容: {answer2}")
        