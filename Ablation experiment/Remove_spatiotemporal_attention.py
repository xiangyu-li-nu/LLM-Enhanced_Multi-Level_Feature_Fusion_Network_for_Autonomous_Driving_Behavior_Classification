import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm

# Check if a GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 1. Load the data
df = pd.read_csv('extracted_features_with_analysis.csv', encoding='latin1')

# 2. Extract necessary columns
text_col = 'analysis'  # Text column
label_col = 'label'  # Label column
numerical_features = [  # Numerical features for the model
    'acceleration_autocorrelation', 'acceleration_change_rate', 'acceleration_jerk_cross_correlation',
    'acceleration_kurtosis', 'acceleration_max', 'acceleration_mean', 'acceleration_median',
    'acceleration_min', 'acceleration_quantile25', 'acceleration_quantile75', 'acceleration_skewness',
    'acceleration_std', 'jerk_kurtosis', 'jerk_max', 'jerk_mean', 'jerk_median', 'jerk_min',
    'jerk_quantile25', 'jerk_quantile75', 'jerk_skewness', 'jerk_std', 'num_hard_accelerations',
    'num_hard_brakes', 'num_hard_turns', 'speed_acceleration_cross_correlation', 'speed_autocorrelation',
    'speed_change_rate', 'speed_kurtosis', 'speed_max', 'speed_mean', 'speed_median', 'speed_min',
    'speed_quantile25', 'speed_quantile75', 'speed_skewness', 'speed_std'
]

texts = df[text_col].astype(str).tolist()  # Convert text column to a list of strings
labels = df[label_col].astype(str).tolist()  # Convert label column to a list of strings
numerical_data = df[numerical_features].values  # Extract numerical feature values

# 3. Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(label_encoder.classes_)
print(f"Labels: {label_encoder.classes_}")

# 4. Standardize numerical features
scaler = StandardScaler()
numerical_data = scaler.fit_transform(numerical_data)

# 5. Split data into training and test sets
train_texts, test_texts, train_labels, test_labels, train_num, test_num = train_test_split(
    texts, encoded_labels, numerical_data, test_size=0.2, random_state=42, stratify=encoded_labels
)

# 6. Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 7. Define custom dataset class
class MultiModalDataset(Dataset):
    def __init__(self, texts, numerical_features, labels, tokenizer, max_length=256):
        self.texts = texts  # Text data
        self.numerical_features = numerical_features  # Numerical feature data
        self.labels = labels  # Labels
        self.tokenizer = tokenizer  # Tokenizer for text processing
        self.max_length = max_length  # Maximum token length for padding/truncating

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]  # Get text at index idx
        numerical = self.numerical_features[idx]  # Get numerical features at index idx
        label = self.labels[idx]  # Get label at index idx

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add special tokens like [CLS] and [SEP]
            max_length=self.max_length,  # Pad or truncate to the specified max length
            padding='max_length',  # Pad to max length
            truncation=True,  # Truncate if necessary
            return_attention_mask=True,  # Return attention mask for padding
            return_tensors='pt',  # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),  # Flatten input_ids tensor
            'attention_mask': encoding['attention_mask'].flatten(),  # Flatten attention_mask tensor
            'numerical': torch.tensor(numerical, dtype=torch.float),  # Convert numerical features to tensor
            'labels': torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        }

# 8. Create datasets and data loaders
batch_size = 64
num_epochs = 10  # You can modify this value as needed

train_dataset = MultiModalDataset(train_texts, train_num, train_labels, tokenizer)
test_dataset = MultiModalDataset(test_texts, test_num, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 10. Define the multi-modal classifier (remove spatiotemporal attention mechanism)
class MultiModalClassifier(nn.Module):
    def __init__(self, num_numerical_features, num_labels, roberta_model_name='roberta-base'):
        super(MultiModalClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)  # Initialize RoBERTa model
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer

        # Numerical features branch: multi-scale 1D convolution (remove spatiotemporal attention mechanism)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3)
        self.relu = nn.ReLU()  # ReLU activation function
        self.pool = nn.AdaptiveMaxPool1d(1)  # Max pooling

        # Deeper convolution processing
        self.num_conv = nn.Sequential(
            nn.Conv1d(in_channels=64*3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        # Fully connected layers for processing convolutional features
        self.num_fc = nn.Sequential(
            nn.Linear(128, 256),  # Fully connected layer
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Add a linear layer to map text features from 768 to 128 dimensions
        self.text_fc = nn.Linear(self.roberta.config.hidden_size, 128)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 256),  # 128 (text) + 128 (numerical) = 256
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, numerical):
        # Text branch
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)

        # Map text features to 128 dimensions
        text_output = self.text_fc(pooled_output)  # [batch_size, 128]

        # Numerical features branch
        numerical = numerical.unsqueeze(1)  # [batch_size, 1, num_features]

        conv1_out = self.relu(self.conv1(numerical))  # [batch_size, 64, num_features]
        conv2_out = self.relu(self.conv2(numerical))  # [batch_size, 64, num_features]
        conv3_out = self.relu(self.conv3(numerical))  # [batch_size, 64, num_features]

        # Concatenate multi-scale features
        combined_conv = torch.cat((conv1_out, conv2_out, conv3_out), dim=1)  # [batch_size, 192, num_features]

        # Apply deeper convolutional processing
        combined_conv = self.num_conv(combined_conv)  # [batch_size, 128, 1]
        combined_conv = combined_conv.squeeze(-1)  # [batch_size, 128]

        # Fully connected layer processing
        num_output = self.num_fc(combined_conv)  # [batch_size, 128]

        # Feature fusion (direct concatenation)
        fused_features = torch.cat((text_output, num_output), dim=1)  # [batch_size, 256]

        # Classification
        logits = self.classifier(fused_features)  # [batch_size, num_labels]
        return logits

# 11. Initialize the model
model = MultiModalClassifier(num_numerical_features=len(numerical_features), num_labels=num_labels)
model = model.to(device)  # Move the model to the specified device (GPU or CPU)

# 12. Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)  # AdamW optimizer with learning rate 2e-5
criterion = nn.CrossEntropyLoss()  # Cross entropy loss function for classification

# 13. Training function
def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()  # Set the model to training mode
    total_loss = 0  # Variable to accumulate the loss during training
    for batch in tqdm(dataloader, desc="Training"):  # Iterate over the batches in the dataloader
        optimizer.zero_grad()  # Reset gradients to zero before each backward pass
        input_ids = batch['input_ids'].to(device)  # Move input_ids to the device (GPU/CPU)
        attention_mask = batch['attention_mask'].to(device)  # Move attention_mask to the device
        numerical = batch['numerical'].to(device)  # Move numerical features to the device
        labels = batch['labels'].to(device)  # Move labels to the device

        logits = model(input_ids, attention_mask, numerical)  # Get the model predictions
        loss = criterion(logits, labels)  # Calculate the loss
        total_loss += loss.item()  # Add the loss for this batch to the total loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters with the optimizer
    avg_loss = total_loss / len(dataloader)  # Calculate the average loss for the epoch
    print(f"Average training loss: {avg_loss:.4f}")  # Print the average training loss

# 14. Evaluation function
def evaluate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []  # List to store predicted labels
    true_labels = []  # List to store true labels

    with torch.no_grad():  # Disable gradient computation during evaluation
        for batch in tqdm(dataloader, desc="Evaluating"):  # Iterate over the batches in the dataloader
            input_ids = batch['input_ids'].to(device)  # Move input_ids to the device
            attention_mask = batch['attention_mask'].to(device)  # Move attention_mask to the device
            numerical = batch['numerical'].to(device)  # Move numerical features to the device
            labels = batch['labels'].to(device)  # Move labels to the device
            logits = model(input_ids, attention_mask, numerical)  # Get the model predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # Get the predicted labels
            labels = labels.cpu().numpy()  # Get the true labels
            predictions.extend(preds)  # Add predictions to the list
            true_labels.extend(labels)  # Add true labels to the list
    return true_labels, predictions  # Return true labels and predictions

# 15. Train and evaluate the model
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_epoch(model, train_loader, optimizer, device, criterion)  # Train for one epoch
    true_labels, predictions = evaluate(model, test_loader, device)  # Evaluate on the test set
    report = classification_report(
        true_labels,
        predictions,
        target_names=label_encoder.classes_, digits=4  # Generate a classification report with 4 decimal places
    )
    print("Classification Report:")
    print(report)  # Print the classification report

# 16. Save the model (optional)
torch.save(model.state_dict(), 'multi_modal_roberta_classifier_no_attention.pth')  # Save the model state_dict
print("Model saved as 'multi_modal_roberta_classifier_no_attention.pth'")  # Print confirmation of saving
