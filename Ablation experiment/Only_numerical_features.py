
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
# Check if there is a GPU available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 1. Read data
df = pd.read_csv('extracted_features_with_analysis.csv', encoding='latin1')

# 2. Extract necessary columns
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

labels = df[label_col].astype(str).tolist()
numerical_data = df[numerical_features].values

# 3. Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(label_encoder.classes_)
print(f"Labels: {label_encoder.classes_}")

# 4. Standardize numerical features
scaler = StandardScaler()
numerical_data = scaler.fit_transform(numerical_data)

# 5. Split data into training and testing sets
train_num, test_num, train_labels, test_labels = train_test_split(
    numerical_data, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# 6. Define custom dataset
class NumericalDataset(Dataset):
    def __init__(self, numerical_features, labels):
        self.numerical_features = numerical_features
        self.labels = labels

    def __len__(self):
        return len(self.numerical_features)

    def __getitem__(self, idx):
        numerical = self.numerical_features[idx]
        label = self.labels[idx]
        return {
            'numerical': torch.tensor(numerical, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 7. Create dataset and data loader
batch_size = 64
num_epochs = 100  # You can modify this value as needed

train_dataset = NumericalDataset(train_num, train_labels)
test_dataset = NumericalDataset(test_num, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 8. Define attention modules
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        attention = self.sigmoid(out)
        return x * attention.expand_as(x)

class SpatioTemporalAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(SpatioTemporalAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# 9. Define model (only using numerical features)
class NumericalClassifier(nn.Module):
    def __init__(self, num_numerical_features, num_labels):
        super(NumericalClassifier, self).__init__()

        # Numerical feature branch: multi-scale 1D convolution + spatio-temporal attention mechanism + deeper convolution network
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Spatio-temporal attention mechanism
        self.spatio_temporal_attention = SpatioTemporalAttention(in_channels=64*3, reduction=16, kernel_size=7)

        # Deeper convolution processing
        self.num_conv = nn.Sequential(
            nn.Conv1d(in_channels=64*3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        # Fully connected layers to process convolution features
        self.num_fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, numerical):
        # Numerical feature branch
        numerical = numerical.unsqueeze(1)  # [batch_size, 1, num_features]

        conv1_out = self.relu(self.conv1(numerical))  # [batch_size, 64, num_features]
        conv2_out = self.relu(self.conv2(numerical))  # [batch_size, 64, num_features]
        conv3_out = self.relu(self.conv3(numerical))  # [batch_size, 64, num_features]

        # Concatenate multi-scale features
        combined_conv = torch.cat((conv1_out, conv2_out, conv3_out), dim=1)  # [batch_size, 192, num_features]

        # Spatio-temporal attention mechanism
        combined_conv = self.spatio_temporal_attention(combined_conv)  # [batch_size, 192, num_features]

        # Deeper convolution processing
        combined_conv = self.num_conv(combined_conv)  # [batch_size, 128, 1]
        combined_conv = combined_conv.squeeze(-1)  # [batch_size, 128]

        # Fully connected layers processing
        num_output = self.num_fc(combined_conv)  # [batch_size, 128]

        # Classification
        logits = self.classifier(num_output)  # [batch_size, num_labels]
        return logits

# 10. Initialize the model
model = NumericalClassifier(num_numerical_features=len(numerical_features), num_labels=num_labels)
model = model.to(device)

# 11. Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 12. Training function
def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        numerical = batch['numerical'].to(device)
        labels = batch['labels'].to(device)

        logits = model(numerical)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(dataloader)
    print(f"Average training loss: {avg_loss:.4f}")

# 13. Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            numerical = batch['numerical'].to(device)
            labels = batch['labels'].to(device)
            logits = model(numerical)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels)
    return true_labels, predictions

# 14. Training and evaluation
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_epoch(model, train_loader, optimizer, device, criterion)
    true_labels, predictions = evaluate(model, test_loader, device)
    report = classification_report(
        true_labels,
        predictions,
        target_names=label_encoder.classes_, digits=4
    )
    print("Classification Report:")
    print(report)

# 15. Save model (optional)
torch.save(model.state_dict(), 'numerical_only_classifier.pth')
print("Model has been saved as 'numerical_only_classifier.pth'")

