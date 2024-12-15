import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

# Check if a GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 1. Read the data
df = pd.read_csv('extracted_features_with_analysis.csv', encoding='latin1')

# 2. Extract the required columns
text_col = 'analysis'
label_col = 'label'

texts = df[text_col].astype(str).tolist()
labels = df[label_col].astype(str).tolist()

# 3. Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(label_encoder.classes_)
print(f"Labels: {label_encoder.classes_}")

# 4. Split the data into training and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# 5. Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 6. Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 7. Create datasets and dataloaders
batch_size = 64
num_epochs = 10  # You can adjust this value as needed

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 8. Define the model (using only text features)
class TextClassifier(nn.Module):
    def __init__(self, num_labels, roberta_model_name='roberta-base'):
        super(TextClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.dropout = nn.Dropout(p=0.3)
        # Add a linear layer to map text features from 768 dimensions to 256 dimensions
        self.fc = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        # Text branch
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)  # [batch_size, num_labels]
        return logits

# 9. Initialize the model
model = TextClassifier(num_labels=num_labels)
model = model.to(device)

# 10. Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 11. Training function
def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(dataloader)
    print(f"Average training loss: {avg_loss:.4f}")

# 12. Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels)
    return true_labels, predictions

# 13. Train and evaluate
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

# 14. Save the model (optional)
torch.save(model.state_dict(), 'text_only_roberta_classifier.pth')
print("Model saved as 'text_only_roberta_classifier.pth'")
