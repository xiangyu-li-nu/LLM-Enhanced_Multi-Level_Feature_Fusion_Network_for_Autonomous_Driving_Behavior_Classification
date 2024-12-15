import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style for better aesthetics
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 1. Load Data
df = pd.read_csv('extracted_features_with_analysis.csv', encoding='latin1')

# 2. Extract necessary columns
text_col = 'analysis'  # Text column for analysis
label_col = 'label'  # Label column for classification
numerical_features = [  # List of numerical features for analysis
    'acceleration_autocorrelation', 'acceleration_change_rate', 'acceleration_jerk_cross_correlation',
    'acceleration_kurtosis', 'acceleration_max', 'acceleration_mean', 'acceleration_median',
    'acceleration_min', 'acceleration_quantile25', 'acceleration_quantile75', 'acceleration_skewness',
    'acceleration_std', 'jerk_kurtosis', 'jerk_max', 'jerk_mean', 'jerk_median', 'jerk_min',
    'jerk_quantile25', 'jerk_quantile75', 'jerk_skewness', 'jerk_std', 'num_hard_accelerations',
    'num_hard_brakes', 'num_hard_turns', 'speed_acceleration_cross_correlation', 'speed_autocorrelation',
    'speed_change_rate', 'speed_kurtosis', 'speed_max', 'speed_mean', 'speed_median', 'speed_min',
    'speed_quantile25', 'speed_quantile75', 'speed_skewness', 'speed_std'
]

texts = df[text_col].astype(str).tolist()  # Convert text data to list of strings
labels = df[label_col].astype(str).tolist()  # Convert labels to list of strings
numerical_data = df[numerical_features].values  # Extract numerical features as numpy array

# 3. Encode labels
label_encoder = LabelEncoder()  # Initialize label encoder
encoded_labels = label_encoder.fit_transform(labels)  # Encode labels into numerical format
num_labels = len(label_encoder.classes_)  # Get the number of unique labels
print(f"Labels: {label_encoder.classes_}")

# 4. Standardize numerical features
scaler = StandardScaler()  # Initialize the standard scaler
numerical_data = scaler.fit_transform(numerical_data)  # Standardize the numerical data

# 5. Split into training and testing sets
train_texts, test_texts, train_labels, test_labels, train_num, test_num = train_test_split(
    texts, encoded_labels, numerical_data, test_size=0.2, random_state=42, stratify=encoded_labels
)  # Split data into training and testing sets (80-20 split)

# 6. Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # Load pre-trained Roberta tokenizer

# 7. Define custom dataset
class MultiModalDataset(Dataset):
    def __init__(self, texts, numerical_features, labels, tokenizer, max_length=256):
        self.texts = texts  # Text data
        self.numerical_features = numerical_features  # Numerical data
        self.labels = labels  # Labels
        self.tokenizer = tokenizer  # Tokenizer for text processing
        self.max_length = max_length  # Maximum length for tokenized text

    def __len__(self):
        return len(self.texts)  # Return the size of the dataset

    def __getitem__(self, idx):
        text = self.texts[idx]  # Get text at index `idx`
        numerical = self.numerical_features[idx]  # Get numerical features at index `idx`
        label = self.labels[idx]  # Get label at index `idx`

        # Tokenize the text using the Roberta tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add special tokens (e.g., [CLS], [SEP])
            max_length=self.max_length,  # Set maximum token length
            padding='max_length',  # Pad sequences to the same length
            truncation=True,  # Truncate longer sequences
            return_attention_mask=True,  # Return attention mask
            return_tensors='pt',  # Return PyTorch tensors
        )

        # Return tokenized inputs and numerical data
        return {
            'input_ids': encoding['input_ids'].flatten(),  # Flatten input IDs (tokenized text)
            'attention_mask': encoding['attention_mask'].flatten(),  # Flatten attention mask
            'numerical': torch.tensor(numerical, dtype=torch.float),  # Convert numerical data to tensor
            'labels': torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        }


# 8. Create datasets and dataloaders
batch_size = 64  # Batch size for training and testing
num_epochs = 20  # Adjust as needed (number of epochs for training)

train_dataset = MultiModalDataset(train_texts, train_num, train_labels, tokenizer)  # Create the training dataset
test_dataset = MultiModalDataset(test_texts, test_num, test_labels, tokenizer)  # Create the testing dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Create the training dataloader
test_loader = DataLoader(test_dataset, batch_size=batch_size)  # Create the testing dataloader

# 9. Define attention modules
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Adaptive average pooling
        self.fc = nn.Sequential(  # Fully connected layers for channel attention
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()  # Sigmoid activation for attention weight
        )

    def forward(self, x):
        b, c, _ = x.size()  # Get batch size, channels, and sequence length
        y = self.avg_pool(x).view(b, c)  # Apply average pooling
        y = self.fc(y).view(b, c, 1)  # Apply fully connected layers
        return x * y.expand_as(x)  # Multiply the input with attention weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2  # Calculate padding
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)  # 1D convolution layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for spatial attention

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling along the channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling along the channel dimension
        concat = torch.cat([avg_out, max_out], dim=1)  # Concatenate average and max pooled features
        out = self.conv(concat)  # Apply convolution
        attention = self.sigmoid(out)  # Apply sigmoid activation
        return x * attention.expand_as(x)  # Multiply the input with attention weights

class SpatioTemporalAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(SpatioTemporalAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)  # Channel attention
        self.spatial_attention = SpatialAttention(kernel_size)  # Spatial attention

    def forward(self, x):
        x = self.channel_attention(x)  # Apply channel attention
        x = self.spatial_attention(x)  # Apply spatial attention
        return x  # Return the attended features

# 10. Define the multimodal model
class MultiModalClassifier(nn.Module):
    def __init__(self, num_numerical_features, num_labels, roberta_model_name='roberta-base'):
        super(MultiModalClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)  # Load pre-trained RoBERTa model
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer

        # Numerical features branch: Multi-scale convolution + Spatio-temporal attention + Deeper convolutional network
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)  # Convolutional layer 1
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)  # Convolutional layer 2
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3)  # Convolutional layer 3
        self.relu = nn.ReLU()  # ReLU activation function
        self.pool = nn.AdaptiveMaxPool1d(1)  # Adaptive max pooling layer

        # Spatio-temporal attention mechanism
        self.spatio_temporal_attention = SpatioTemporalAttention(in_channels=64*3, reduction=16, kernel_size=7)

        # Deeper convolutional processing
        self.num_conv = nn.Sequential(
            nn.Conv1d(in_channels=64*3, out_channels=128, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # Convolutional layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # Adaptive max pooling layer
        )

        # Fully connected layers for processed numerical features
        self.num_fc = nn.Sequential(
            nn.Linear(128, 256),  # Linear layer
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Linear layer to map text features from 768 to 128 dimensions
        self.text_fc = nn.Linear(self.roberta.config.hidden_size, 128)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 256),  # 128 (text) + 128 (numerical) = 256
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_labels)  # Output layer with number of labels
        )

    def forward(self, input_ids, attention_mask, numerical):
        # Text branch
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)  # Get RoBERTa outputs
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size] (pooled representation)
        pooled_output = self.dropout(pooled_output)

        # Map text features to 128 dimensions
        text_output = self.text_fc(pooled_output)  # [batch_size, 128]

        # Numerical features branch
        numerical = numerical.unsqueeze(1)  # [batch_size, 1, num_features]

        # Apply convolutions to numerical features
        conv1_out = self.relu(self.conv1(numerical))  # [batch_size, 64, num_features]
        conv2_out = self.relu(self.conv2(numerical))  # [batch_size, 64, num_features]
        conv3_out = self.relu(self.conv3(numerical))  # [batch_size, 64, num_features]

        # Concatenate multi-scale features
        combined_conv = torch.cat((conv1_out, conv2_out, conv3_out), dim=1)  # [batch_size, 192, num_features]

        # Apply spatio-temporal attention mechanism
        combined_conv = self.spatio_temporal_attention(combined_conv)  # [batch_size, 192, num_features]

        # Apply deeper convolutional processing
        combined_conv = self.num_conv(combined_conv)  # [batch_size, 128, 1]
        combined_conv = combined_conv.squeeze(-1)  # [batch_size, 128]

        # Fully connected layers for numerical features
        num_output = self.num_fc(combined_conv)  # [batch_size, 128]

        # Feature fusion (concatenation)
        fused_features = torch.cat((text_output, num_output), dim=1)  # [batch_size, 256]

        # Classification
        logits = self.classifier(fused_features)  # [batch_size, num_labels]
        return logits  # Return the class logits

# 11. Initialize the model
model = MultiModalClassifier(num_numerical_features=len(numerical_features), num_labels=num_labels)  # Initialize the multimodal model
model = model.to(device)  # Move the model to the specified device (GPU or CPU)

# 12. Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)  # AdamW optimizer with a learning rate of 2e-5
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss function for classification

# 13. Training function
def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()  # Set model to training mode
    total_loss = 0  # Initialize total loss
    correct_predictions = 0  # Initialize the count of correct predictions
    for batch in tqdm(dataloader, desc="Training"):  # Iterate through the training batches
        optimizer.zero_grad()  # Zero the gradients to prevent accumulation
        input_ids = batch['input_ids'].to(device)  # Move input_ids to the device
        attention_mask = batch['attention_mask'].to(device)  # Move attention_mask to the device
        numerical = batch['numerical'].to(device)  # Move numerical data to the device
        labels = batch['labels'].to(device)  # Move labels to the device

        logits = model(input_ids, attention_mask, numerical)  # Get model predictions (logits)
        loss = criterion(logits, labels)  # Calculate the loss
        total_loss += loss.item()  # Add the loss to total_loss
        _, preds = torch.max(logits, dim=1)  # Get the predicted class labels
        correct_predictions += torch.sum(preds == labels)  # Count correct predictions

        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters
    avg_loss = total_loss / len(dataloader)  # Calculate the average loss for the epoch
    avg_acc = correct_predictions.double() / len(dataloader.dataset)  # Calculate the average accuracy for the epoch
    return avg_loss, avg_acc  # Return the average loss and accuracy

# 14. Evaluation function
def evaluate(model, dataloader, device, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize total loss
    correct_predictions = 0  # Initialize the count of correct predictions
    predictions = []  # Initialize list for storing predictions
    true_labels = []  # Initialize list for storing true labels

    with torch.no_grad():  # No gradient calculation during evaluation
        for batch in tqdm(dataloader, desc="Evaluating"):  # Iterate through the test batches
            input_ids = batch['input_ids'].to(device)  # Move input_ids to the device
            attention_mask = batch['attention_mask'].to(device)  # Move attention_mask to the device
            numerical = batch['numerical'].to(device)  # Move numerical data to the device
            labels = batch['labels'].to(device)  # Move labels to the device

            logits = model(input_ids, attention_mask, numerical)  # Get model predictions (logits)
            loss = criterion(logits, labels)  # Calculate the loss
            total_loss += loss.item()  # Add the loss to total_loss

            _, preds = torch.max(logits, dim=1)  # Get the predicted class labels
            correct_predictions += torch.sum(preds == labels)  # Count correct predictions

            predictions.extend(preds.cpu().numpy())  # Store predictions in a list
            true_labels.extend(labels.cpu().numpy())  # Store true labels in a list

    avg_loss = total_loss / len(dataloader)  # Calculate the average loss for the evaluation
    avg_acc = correct_predictions.double() / len(dataloader.dataset)  # Calculate the average accuracy
    return avg_loss, avg_acc, true_labels, predictions  # Return average loss, accuracy, true labels, and predictions

# 15. Training and evaluation
train_losses = []  # List to store training losses
train_accuracies = []  # List to store training accuracies
val_losses = []  # List to store validation losses
val_accuracies = []  # List to store validation accuracies

for epoch in range(num_epochs):  # Loop over epochs
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion)  # Train for one epoch
    val_loss, val_acc, true_labels, predictions = evaluate(model, test_loader, device, criterion)  # Evaluate the model

    train_losses.append(train_loss)  # Append training loss for this epoch
    train_accuracies.append(train_acc.item())  # Append training accuracy for this epoch
    val_losses.append(val_loss)  # Append validation loss for this epoch
    val_accuracies.append(val_acc.item())  # Append validation accuracy for this epoch

    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# 16. Plot training loss and accuracy curves
epochs = range(1, num_epochs + 1)  # Define the range of epochs for plotting

plt.figure(figsize=(14, 5))  # Create a figure with specific size

# Loss curve
plt.subplot(1, 2, 1)  # Plot the loss curve
plt.plot(epochs, train_losses, 'b-', label='Training Loss')  # Training loss (blue line)
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')  # Validation loss (red line)
plt.title('Training and Validation Loss')  # Title of the loss plot
plt.xlabel('Epochs')  # X-axis label
plt.ylabel('Loss')  # Y-axis label
plt.legend()  # Display legend

# Accuracy curve
plt.subplot(1, 2, 2)  # Plot the accuracy curve
plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')  # Training accuracy (blue line)
plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')  # Validation accuracy (red line)
plt.title('Training and Validation Accuracy')  # Title of the accuracy plot
plt.xlabel('Epochs')  # X-axis label
plt.ylabel('Accuracy')  # Y-axis label
plt.legend()  # Display legend

plt.tight_layout()  # Adjust layout for better spacing
plt.savefig('training_curves.png', dpi=300)  # Save the plot as a PNG file with high resolution
plt.show()  # Display the plot

# 17. Classification report and confusion matrix
report = classification_report(
    true_labels,  # True labels
    predictions,  # Predicted labels
    target_names=label_encoder.classes_,  # Class names
    digits=4  # Precision of 4 decimal places
)
print("\nClassification Report:")
print(report)  # Print the classification report

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)  # Compute confusion matrix
plt.figure(figsize=(8, 6))  # Create a figure for the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)  # Display confusion matrix
disp.plot(cmap='Blues', values_format='d')  # Plot confusion matrix with blue color scheme
plt.title('Confusion Matrix')  # Title of the confusion matrix plot
plt.savefig('confusion_matrix.png', dpi=300)  # Save the confusion matrix plot
plt.show()  # Display the plot

# 18. Save the model (optional)
torch.save(model.state_dict(), 'enhanced_multi_modal_roberta_classifier.pth')  # Save the model state dict
print("Model saved as 'enhanced_multi_modal_roberta_classifier.pth'")  # Print confirmation message
