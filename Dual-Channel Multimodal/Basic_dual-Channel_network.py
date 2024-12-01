import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm

# 检查是否有可用的GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 1. 读取数据
df = pd.read_csv('extracted_features_with_analysis.csv')

# 2. 提取需要的列
text_col = 'analysis'
label_col = 'label'
numerical_features = [
    'acceleration_autocorrelation', 'acceleration_change_rate', 'acceleration_jerk_cross_correlation',
    'acceleration_kurtosis', 'acceleration_max', 'acceleration_mean', 'acceleration_median',
    'acceleration_min', 'acceleration_quantile25', 'acceleration_quantile75', 'acceleration_skewness',
    'acceleration_std', 'jerk_kurtosis', 'jerk_max', 'jerk_mean', 'jerk_median', 'jerk_min',
    'jerk_quantile25', 'jerk_quantile75', 'jerk_skewness', 'jerk_std', 'num_hard_accelerations',
    'num_hard_brakes', 'num_hard_turns', 'speed_acceleration_cross_correlation', 'speed_autocorrelation',
    'speed_change_rate', 'speed_kurtosis', 'speed_max', 'speed_mean', 'speed_median', 'speed_min',
    'speed_quantile25', 'speed_quantile75', 'speed_skewness', 'speed_std'
]

texts = df[text_col].astype(str).tolist()
labels = df[label_col].astype(str).tolist()
numerical_data = df[numerical_features].values

# 3. 编码标签
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(label_encoder.classes_)
print(f"Labels: {label_encoder.classes_}")

# 4. 标准化数值特征
scaler = StandardScaler()
numerical_data = scaler.fit_transform(numerical_data)

# 5. 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels, train_num, test_num = train_test_split(
    texts, encoded_labels, numerical_data, test_size=0.2, random_state=42, stratify=encoded_labels
)

# 6. 初始化分词器
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# 7. 定义自定义数据集
class MultiModalDataset(Dataset):
    def __init__(self, texts, numerical_features, labels, tokenizer, max_length=256):
        self.texts = texts
        self.numerical_features = numerical_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        numerical = self.numerical_features[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical': torch.tensor(numerical, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 8. 创建数据集和数据加载器
batch_size = 64
# 14. 设置训练轮数
num_epochs = 10  # 你可以根据需要修改这个值

train_dataset = MultiModalDataset(train_texts, train_num, train_labels, tokenizer)
test_dataset = MultiModalDataset(test_texts, test_num, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 9. 定义多模态模型（数值特征分支使用多尺度卷积）
class MultiModalClassifier(nn.Module):
    def __init__(self, num_numerical_features, num_labels, roberta_model_name='roberta-base'):
        super(MultiModalClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.dropout = nn.Dropout(p=0.3)

        # 数值特征分支：多尺度1D卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

        # 全连接层用于处理卷积后的特征
        self.num_fc = nn.Sequential(
            nn.Linear(32 * 3, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, numerical):
        # 文本分支
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)

        # 数值特征分支
        # 先调整维度以适应Conv1d: [batch_size, 1, num_features]
        numerical = numerical.unsqueeze(1)

        conv1_out = self.relu(self.conv1(numerical))  # [batch_size, 32, num_features]
        conv2_out = self.relu(self.conv2(numerical))  # [batch_size, 32, num_features]
        conv3_out = self.relu(self.conv3(numerical))  # [batch_size, 32, num_features]

        # 池化
        pool1 = self.pool(conv1_out).squeeze(-1)  # [batch_size, 32]
        pool2 = self.pool(conv2_out).squeeze(-1)  # [batch_size, 32]
        pool3 = self.pool(conv3_out).squeeze(-1)  # [batch_size, 32]

        # 拼接多尺度特征
        combined_conv = torch.cat((pool1, pool2, pool3), dim=1)  # [batch_size, 96]

        # 全连接层处理
        num_output = self.num_fc(combined_conv)  # [batch_size, 64]

        # 合并
        combined = torch.cat((pooled_output, num_output), dim=1)  # [batch_size, hidden_size + 64]

        # 分类
        logits = self.classifier(combined)
        return logits


# 10. 初始化模型
model = MultiModalClassifier(num_numerical_features=len(numerical_features), num_labels=num_labels)
model = model.to(device)

# 11. 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()


# 12. 训练函数
def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical = batch['numerical'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask, numerical)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(dataloader)
    print(f"Average training loss: {avg_loss}")


# 13. 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical = batch['numerical'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, numerical)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    return true_labels, predictions



# 15. 训练和评估
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_epoch(model, train_loader, optimizer, device, criterion)
    true_labels, predictions = evaluate(model, test_loader, device)
    report = classification_report(
        true_labels,
        predictions,
        target_names=label_encoder.classes_
    )
    print("Classification Report:")
    print(report)

# 16. 保存模型（可选）
torch.save(model.state_dict(), 'multi_modal_roberta_classifier.pth')
print("模型已保存为 'multi_modal_roberta_classifier.pth'")
