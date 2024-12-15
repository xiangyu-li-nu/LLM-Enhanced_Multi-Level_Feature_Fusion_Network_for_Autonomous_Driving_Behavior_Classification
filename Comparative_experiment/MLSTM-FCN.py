import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, GlobalAveragePooling1D, concatenate, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('extracted_features.csv')

# Separate features and labels
X = df.drop(columns=['label'])
y = df['label']

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Assume each sample has 12 time steps, each with 3 features (total 36 features)
# You can adjust based on actual feature count and time steps
time_steps = 12
features = X_scaled.shape[1] // time_steps

# Raise an exception if feature count is not divisible by time steps
if X_scaled.shape[1] % time_steps != 0:
    raise ValueError(f"Feature count must be divisible by time steps. Current feature count: {X_scaled.shape[1]}, Time steps: {time_steps}")

# Reshape features to fit LSTM and Conv1D layers
X_scaled = X_scaled.reshape((X_scaled.shape[0], time_steps, features))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Build the MLSTM-FCN model
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# LSTM branch
lstm_branch = LSTM(128, return_sequences=True)(input_layer)
lstm_branch = BatchNormalization()(lstm_branch)
lstm_branch = Dropout(0.5)(lstm_branch)
lstm_branch = LSTM(64, return_sequences=False)(lstm_branch)
lstm_branch = BatchNormalization()(lstm_branch)
lstm_branch = Dropout(0.5)(lstm_branch)

# FCN branch
fcn_branch = Conv1D(filters=128, kernel_size=8, activation='relu', padding='same')(input_layer)
fcn_branch = BatchNormalization()(fcn_branch)
fcn_branch = Dropout(0.5)(fcn_branch)
fcn_branch = Conv1D(filters=256, kernel_size=5, activation='relu', padding='same')(fcn_branch)
fcn_branch = BatchNormalization()(fcn_branch)
fcn_branch = Dropout(0.5)(fcn_branch)
fcn_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(fcn_branch)
fcn_branch = BatchNormalization()(fcn_branch)
fcn_branch = GlobalAveragePooling1D()(fcn_branch)

# Merge LSTM and FCN branches
merged = concatenate([lstm_branch, fcn_branch])

# Fully connected layer
dense = Dense(128, activation='relu')(merged)
dense = Dropout(0.5)(dense)

# Output layer
output = Dense(y_train.shape[1], activation='softmax')(dense)

# Define the model
model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,  # Adjust epochs as needed
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert predictions back to class labels
y_train_pred_classes = np.argmax(y_train_pred, axis=1)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_train_true_classes = np.argmax(y_train, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

# Generate classification report
print('Classification Report - Training Set:\n')
print(classification_report(
    y_train_true_classes,
    y_train_pred_classes,
    target_names=label_encoder.classes_,
    digits=4
))
print('-' * 80)
print('Classification Report - Test Set:\n')
print(classification_report(
    y_test_true_classes,
    y_test_pred_classes,
    target_names=label_encoder.classes_,
    digits=4
))
print('-' * 80)

# Plot training and validation loss and accuracy curves
plt.figure(figsize=(12, 4))

# Plot loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
