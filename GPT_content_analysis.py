import pandas as pd
from tqdm import tqdm
import time
from openai import OpenAI
import openai
import os

# 配置 OpenAI 客户端
client = OpenAI(api_key='')  # 替换为您的实际API密钥
client.base_url = ''

# 设置API请求的基本参数
MODEL = "gpt-4-turbo-2024-04-09"  # 请根据实际情况选择模型
MAX_RETRIES = 5  # 最大重试次数
SLEEP_TIME = 2  # 重试等待时间（秒）

# 读取CSV文件
input_csv = 'extracted_features.csv'
output_csv = 'extracted_features_with_analysis1.csv'

# 检查输出文件是否已存在，以决定是否写入标题
if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
    start_index = len(df)
else:
    df = pd.read_csv(input_csv)
    df['analysis'] = ""  # 初始化 'analysis' 列
    start_index = 0


# 定义生成分析的函数
def generate_analysis(sample_features):
    # 构建提示词
    prompt = f"""参考以下样本：

### 示例 1: Aggressive
特征值:
acceleration_autocorrelation: 0.498585529
acceleration_change_rate: -0.540906302
acceleration_jerk_cross_correlation: 1.601550615
acceleration_kurtosis: -0.315776162
acceleration_max: -0.0239652
acceleration_mean: -0.179166735
acceleration_median: -0.292816183
acceleration_min: 0.303044767
acceleration_quantile25: -0.223688038
acceleration_quantile75: -0.405500204
acceleration_skewness: 0.667788727
acceleration_std: -0.203233636
jerk_kurtosis: -0.602482772
jerk_max: -0.78015429
jerk_mean: 0.766257494
jerk_median: 0.68334637
jerk_min: 0.60069146
jerk_quantile25: 0.861094559
jerk_quantile75: 0.143505026
jerk_skewness: -0.659453126
jerk_std: -0.629175467
num_hard_accelerations: -0.212182195
num_hard_brakes: -0.183857195
num_hard_turns: -0.238843469
speed_acceleration_cross_correlation: -0.147188622
speed_autocorrelation: 0.18654735
speed_change_rate: -0.417713317
speed_kurtosis: -0.480501285
speed_max: 1.348472565
speed_mean: 1.526594765
speed_median: 1.495037079
speed_min: 1.687192333
speed_quantile25: 1.569356126
speed_quantile75: 1.455720512
speed_skewness: -0.278717445
speed_std: -0.525425996

驾驶风格: Aggressive

分析:
该驾驶员表现出较高的加速度自相关和加速度变化率，说明其驾驶时加速和减速频繁且幅度较大。此外，较高的颠簸相关性和高频率的急加速、急刹车及急转弯，进一步表明其驾驶风格倾向于激进（Aggressive）。速度相关指标显示其在行驶中保持较高且波动较大的速度，综合来看，该驾驶员属于激进型驾驶风格。

---

### 示例 2: Assertive
特征值:
acceleration_autocorrelation: 0.456980946
acceleration_change_rate: -1.305596502
acceleration_jerk_cross_correlation: 2.705796426
acceleration_kurtosis: 0.729887616
acceleration_max: -0.818519101
acceleration_mean: -0.164484406
acceleration_median: -0.173863085
acceleration_min: 0.787273343
acceleration_quantile25: 0.346203663
acceleration_quantile75: -0.695853926
acceleration_skewness: 1.853971468
acceleration_std: -1.231830788
jerk_kurtosis: -0.087828686
jerk_max: -1.154668987
jerk_mean: 0.194641055
jerk_median: 0.036594409
jerk_min: 1.259103006
jerk_quantile25: 1.135796613
jerk_quantile75: -1.065710977
jerk_skewness: 1.334508143
jerk_std: -1.286600963
num_hard_accelerations: -0.212182195
num_hard_brakes: -0.183857195
num_hard_turns: -0.238843469
speed_acceleration_cross_correlation: -0.426618287
speed_autocorrelation: 0.248617076
speed_change_rate: -1.123955575
speed_kurtosis: -0.474456784
speed_max: 0.006827745
speed_mean: 0.253371616
speed_median: 0.257902944
speed_min: 0.48272207
speed_quantile25: 0.370087047
speed_quantile75: 0.13214635
speed_skewness: -0.408477254
speed_std: -0.94394352

驾驶风格: Assertive

分析:
该驾驶员在加速度和速度的自相关性上表现出适度的稳定性，表明其驾驶行为较为自信且果断。加速度变化率较高，显示其在加减速时具有一定的力度。同时，适中的急加速、急刹车及急转弯次数，结合速度保持在合理范围内，综合来看，该驾驶员属于自信型驾驶风格（Assertive）。

---

### 示例 3: Conservative
特征值:
acceleration_autocorrelation: -1.032860758
acceleration_change_rate: -1.387368258
acceleration_jerk_cross_correlation: 0.308294057
acceleration_kurtosis: 1.31851155
acceleration_max: -0.931897999
acceleration_mean: -0.068117109
acceleration_median: -0.075143265
acceleration_min: 0.892082788
acceleration_quantile25: 0.505436753
acceleration_quantile75: -0.632684217
acceleration_skewness: 2.345977543
acceleration_std: -1.382224891
jerk_kurtosis: 2.777818327
jerk_max: -1.276910568
jerk_mean: -0.086761938
jerk_median: 0.058807979
jerk_min: 1.067632482
jerk_quantile25: 1.202496175
jerk_quantile75: -1.204722331
jerk_skewness: -2.545643745
jerk_std: -1.319711788
num_hard_accelerations: -0.212182195
num_hard_brakes: -0.183857195
num_hard_turns: -0.238843469
speed_acceleration_cross_correlation: 1.030213224
speed_autocorrelation: -0.097843201
speed_change_rate: -1.293317034
speed_kurtosis: 1.469883172
speed_max: -1.208458391
speed_mean: -0.949611975
speed_median: -0.920723444
speed_min: -0.705218676
speed_quantile25: -0.797760272
speed_quantile75: -1.067587978
speed_skewness: 1.893022705
speed_std: -1.094323177

驾驶风格: Conservative

分析:
该驾驶员在加速度和速度的各项指标上表现出较低的波动性和较高的稳定性，说明其驾驶行为平稳且谨慎。急加速、急刹车及急转弯次数较少，且速度变化幅度较小，进一步表明其驾驶风格倾向于保守（Conservative）。整体来看，该驾驶员属于保守型驾驶风格。

---

### 示例 4: Moderate
特征值:
acceleration_autocorrelation: -0.172558511
acceleration_change_rate: -1.286579107
acceleration_jerk_cross_correlation: -2.994926832
acceleration_kurtosis: 2.258399278
acceleration_max: -0.97063585
acceleration_mean: -0.098877366
acceleration_median: -0.075358654
acceleration_min: 0.687141737
acceleration_quantile25: 0.492461354
acceleration_quantile75: -0.628252202
acceleration_skewness: -2.612711904
acceleration_std: -1.326882669
jerk_kurtosis: 0.74930306
jerk_max: -1.033801172
jerk_mean: 0.182254421
jerk_median: 0.069326957
jerk_min: 1.136062943
jerk_quantile25: 1.162334435
jerk_quantile75: -1.026060174
jerk_skewness: 1.3074327
jerk_std: -1.246611598
num_hard_accelerations: -0.212182195
num_hard_brakes: -0.183857195
num_hard_turns: -0.238843469
speed_acceleration_cross_correlation: -1.662332424
speed_autocorrelation: -0.089600963
speed_change_rate: -1.286069634
speed_kurtosis: 4.733287142
speed_max: -1.20660776
speed_mean: -0.951207409
speed_median: -0.922349422
speed_min: -0.705218676
speed_quantile25: -0.799408102
speed_quantile75: -1.069177773
speed_skewness: 3.312904374
speed_std: -1.095880206

驾驶风格: Moderate

分析:
该驾驶员在加速度和速度的各项指标上表现出中等水平的波动性，既不极端激进也不过于保守。急加速、急刹车及急转弯次数适中，速度变化较为平稳但偶有波动，显示出其驾驶行为平衡且适度。综合来看，该驾驶员属于中庸型驾驶风格（Moderate）。

---

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
            # 调用OpenAI的ChatCompletion API
            completion = client.chat.completions.create(
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
            analysis = completion.choices[0].message.content.strip()
            return analysis
        except openai.error.OpenAIError as e:
            print(f"OpenAI API request failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Waiting {SLEEP_TIME} seconds before retrying...")
                time.sleep(SLEEP_TIME)
            else:
                print("Maximum retry attempts reached. Skipping this sample.")
                return "Analysis failed"
    return "Analysis failed"


# 添加一个新的列用于存储分析结果
if 'analysis' not in df.columns:
    df['analysis'] = ""

# 使用tqdm显示进度条
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing samples", initial=start_index):
    if index < start_index:
        continue  # 跳过已处理的样本

    # 提取当前样本的特征
    sample_features = row.to_dict()
    # 移除'label'和'analysis'列
    sample_features.pop('label', None)
    sample_features.pop('analysis', None)

    # 生成分析
    analysis = generate_analysis(sample_features)

    print(f"Sample {index}: {analysis}")

    # 保存分析到新的列
    df.at[index, 'analysis'] = analysis

    # 保存当前进度到输出CSV文件
    df.to_csv(output_csv, index=False)

    # 为避免触发API速率限制，适当延时
    time.sleep(1)  # 根据实际情况调整

print("Analysis completed. Results have been saved to 'extracted_features_with_analysis.csv'")
