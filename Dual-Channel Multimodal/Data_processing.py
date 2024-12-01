# import pandas as pd
# import re
#
# # 1. 读取数据
# df = pd.read_csv('extracted_features_with_analysis11.csv', encoding='latin1')
#
# # 2. 提取需要的列
# text_col = 'analysis'
# label_col = 'label'
#
# # 3. 定义需要替换的词汇列表（包含大小写）
# target_words = ['Aggressive', 'Assertive', 'Conservative', 'Moderate']
#
# # 4. 创建正则表达式模式，忽略大小写
# #    \b 确保是完整的单词匹配
# pattern = re.compile(r'\b(' + '|'.join(target_words) + r')\b', re.IGNORECASE)
#
# def replace_with_label(text, label):
#     """
#     替换文本中出现的目标词汇为指定的标签值。
#
#     参数:
#     text (str): 原始文本。
#     label (str): 用于替换的标签值。
#
#     返回:
#     str: 替换后的文本。
#     """
#     return pattern.sub(label, text)
#
# # 5. 应用替换函数到每一行的 'analysis' 列
# df['new_analysis'] = df.apply(lambda row: replace_with_label(row[text_col], row[label_col]), axis=1)
#
# # 6. 保存结果到新的CSV文件
# df.to_csv('extracted_features_with_analysis.csv', index=False, encoding='latin1')
#
# print("替换完成，新的分析已保存到 'extracted_features_with_analysis.csv'。")
import pandas as pd
import re
import os

# 1. 读取数据
# 使用绝对路径确保文件路径正确，请根据实际情况修改路径
file_path = r'extracted_features_with_analysis11.csv'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件不存在：{file_path}")
    exit(1)

try:
    df = pd.read_csv(file_path, encoding='latin1')
    print("文件读取成功，数据预览：")
    print(df.head())
except pd.errors.EmptyDataError:
    print(f"文件为空或没有可解析的列：{file_path}")
    exit(1)
except Exception as e:
    print(f"读取文件时发生错误：{e}")
    exit(1)

# 2. 提取需要的列
text_col = 'analysis'
label_col = 'label'

# 检查必要的列是否存在
if text_col not in df.columns or label_col not in df.columns:
    print(f"缺少必要的列：需要包含 '{text_col}' 和 '{label_col}' 列。")
    exit(1)

# 3. 定义需要删除的词汇列表（包含大小写）
target_words = ['Aggressive', 'Assertive', 'Conservative', 'Moderate']

# 4. 创建正则表达式模式，忽略大小写
#    \b 确保是完整的单词匹配
pattern = re.compile(r'\b(' + '|'.join(target_words) + r')\b', re.IGNORECASE)

def remove_target_words(text):
    """
    删除文本中出现的目标词汇。

    参数:
    text (str): 原始文本。

    返回:
    str: 删除目标词汇后的文本。
    """
    # 使用正则表达式替换目标词汇为空字符串
    cleaned_text = pattern.sub('', text)
    # 去除多余的空格
    cleaned_text = re.sub(' +', ' ', cleaned_text).strip()
    return cleaned_text

# 5. 应用删除函数到每一行的 'analysis' 列
df['new_analysis'] = df[text_col].apply(remove_target_words)

# 6. 保存结果到新的CSV文件
output_path = r'shanchu.csv'

try:
    df.to_csv(output_path, index=False, encoding='latin1')
    print(f"删除完成，新的分析已保存到 '{output_path}'。")
except Exception as e:
    print(f"保存文件时发生错误：{e}")
