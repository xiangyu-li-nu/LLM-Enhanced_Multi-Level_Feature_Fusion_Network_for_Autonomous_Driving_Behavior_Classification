import pandas as pd
from tqdm import tqdm
import time
import openai
import os
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # 获取 API 密钥
# api_key = os.getenv('')
# if not api_key:
#     logging.error("未设置环境变量 OPENAI_API_KEY。请设置后重试。")
#     raise ValueError("请设置环境变量 OPENAI_API_KEY")

# 配置 OpenAI 客户端
openai.api_key = ''
openai.api_base = 'https://api.openai.com/v1'  # 使用官方 API 端点

# 设置 API 请求的基本参数
MODEL = "gpt-4"  # 根据需要选择模型
MAX_RETRIES = 5  # 最大重试次数
SLEEP_TIME = 2  # 重试等待时间（秒）

# 读取 CSV 文件
input_csv = 'extracted_featuresyuanshi.csv'
output_csv = 'extracted_features_with_analysis11.csv'

# 检查输出文件是否已存在，以决定是否写入标题
if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
    start_index = len(df)
    logging.info(f"检测到已存在的输出文件，开始索引为 {start_index}")
else:
    df = pd.read_csv(input_csv)
    df['analysis'] = ""  # 初始化 'analysis' 列
    start_index = 0
    logging.info("未检测到输出文件，初始化新的 DataFrame。")

# 定义生成分析的函数
def generate_analysis(sample_features):
    # 构建提示词
    prompt = f"""参考以下样本：

### 示例 1: Aggressive
特征值:
acceleration_autocorrelation: 0.498585529
acceleration_change_rate: -0.540906302
...
[省略其他示例内容以简洁]

请根据以下特征值分析驾驶风格，并用自然语言描述（100字以内）:

特征值:
acceleration_autocorrelation: {sample_features['acceleration_autocorrelation']}
acceleration_change_rate: {sample_features['acceleration_change_rate']}
acceleration_jerk_cross_correlation: {sample_features['acceleration_jerk_cross_correlation']}
acceleration_kurtosis: {sample_features['acceleration_kurtosis']}
acceleration_max: {sample_features['acceleration_max']}
acceleration_mean: {sample_features['acceleration_mean']}
acceleration_median: {sample_features['acceleration_median']}
acceleration_min: {sample_features['acceleration_min']}
acceleration_quantile25: {sample_features['acceleration_quantile25']}
acceleration_quantile75: {sample_features['acceleration_quantile75']}
acceleration_skewness: {sample_features['acceleration_skewness']}
acceleration_std: {sample_features['acceleration_std']}
jerk_kurtosis: {sample_features['jerk_kurtosis']}
jerk_max: {sample_features['jerk_max']}
jerk_mean: {sample_features['jerk_mean']}
jerk_median: {sample_features['jerk_median']}
jerk_min: {sample_features['jerk_min']}
jerk_quantile25: {sample_features['jerk_quantile25']}
jerk_quantile75: {sample_features['jerk_quantile75']}
jerk_skewness: {sample_features['jerk_skewness']}
jerk_std: {sample_features['jerk_std']}
num_hard_accelerations: {sample_features['num_hard_accelerations']}
num_hard_brakes: {sample_features['num_hard_brakes']}
num_hard_turns: {sample_features['num_hard_turns']}
speed_acceleration_cross_correlation: {sample_features['speed_acceleration_cross_correlation']}
speed_autocorrelation: {sample_features['speed_autocorrelation']}
speed_change_rate: {sample_features['speed_change_rate']}
speed_kurtosis: {sample_features['speed_kurtosis']}
speed_max: {sample_features['speed_max']}
speed_mean: {sample_features['speed_mean']}
speed_median: {sample_features['speed_median']}
speed_min: {sample_features['speed_min']}
speed_quantile25: {sample_features['speed_quantile25']}
speed_quantile75: {sample_features['speed_quantile75']}
speed_skewness: {sample_features['speed_skewness']}
speed_std: {sample_features['speed_std']}

分析 用英文回答:
"""

    for attempt in range(MAX_RETRIES):
        try:
            # 调用 OpenAI 的 ChatCompletion API
            completion = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a driving behavior analyst, skilled in analyzing driving styles based on vehicle sensor data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500,
                n=1,
                stop=None,
            )
            # 确保响应结构正确
            if 'choices' in completion and len(completion['choices']) > 0:
                analysis = completion['choices'][0]['message']['content'].strip()
                return analysis
            else:
                logging.warning(f"Unexpected response structure: {completion}")
                return "Analysis failed: Unexpected response structure"
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API request failed: {e}")
            if attempt < MAX_RETRIES - 1:
                logging.info(f"等待 {SLEEP_TIME} 秒后重试...")
                time.sleep(SLEEP_TIME)
            else:
                logging.error("达到最大重试次数，跳过此样本。")
                return "Analysis failed"
        except Exception as e:
            logging.error(f"发生意外错误: {e}")
            return "Analysis failed"

    return "Analysis failed"

# 添加一个新的列用于存储分析结果
if 'analysis' not in df.columns:
    df['analysis'] = ""
    logging.info("添加 'analysis' 列到 DataFrame。")

# 使用 tqdm 显示进度条
total_samples = df.shape[0]
logging.info(f"开始处理 {total_samples} 个样本。")

for index, row in tqdm(df.iterrows(), total=total_samples, desc="Processing samples", initial=start_index):
    if index < start_index:
        continue  # 跳过已处理的样本

    # 提取当前样本的特征
    sample_features = row.to_dict()
    # 移除 'label' 和 'analysis' 列（如果存在）
    sample_features.pop('label', None)
    sample_features.pop('analysis', None)

    # 生成分析
    analysis = generate_analysis(sample_features)

    logging.info(f"Sample {index}: {analysis}")

    # 保存分析到新的列
    df.at[index, 'analysis'] = analysis

    # 保存当前进度到输出 CSV 文件
    df.to_csv(output_csv, index=False)

    # 为避免触发 API 速率限制，适当延时
    time.sleep(1)  # 根据实际情况调整

logging.info("Analysis completed. Results have been saved to 'extracted_features_with_analysis1.csv'")
