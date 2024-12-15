import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, GlobalAveragePooling1D, concatenate
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

# Assume each sample has 6 time steps, each with 6 features
# 36 = 6 * 6
time_steps = 6
features = X_scaled.shape[1] // time_steps
if X_scaled.shape[1] % time_steps != 0:
    raise ValueError(f"Feature count must be divisible by time steps. Current feature count: {X_scaled.shape[1]}, Time steps: {time_steps}")

# Reshape data to fit the model
X_scaled = X_scaled.reshape((X_scaled.shape[0], time_steps, features))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)

# Build the multi-branch convolutional network (mWDN assumed to be Multi-Branch CNN)
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# Branch 1: Kernel size = 3
conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(input_layer)
conv1 = Dropout(0.5)(conv1)
conv1 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(conv1)
conv1 = Dropout(0.5)(conv1)
conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)
gap1 = GlobalAveragePooling1D()(conv1)

# Branch 2: Kernel size = 5
conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(input_layer)
conv2 = Dropout(0.5)(conv2)
conv2 = Conv1D(filters=256, kernel_size=5, activation='relu', padding='same')(conv2)
conv2 = Dropout(0.5)(conv2)
conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(conv2)
gap2 = GlobalAveragePooling1D()(conv2)

# Branch 3: Kernel size = 7
conv3 = Conv1D(filters=128, kernel_size=7, activation='relu', padding='same')(input_layer)
conv3 = Dropout(0.5)(conv3)
conv3 = Conv1D(filters=256, kernel_size=7, activation='relu', padding='same')(conv3)
conv3 = Dropout(0.5)(conv3)
conv3 = Conv1D(filters=128, kernel_size=7, activation='relu', padding='same')(conv3)
gap3 = GlobalAveragePooling1D()(conv3)

# Merge all branches
merged = concatenate([gap1, gap2, gap3])

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
    epochs=10,  # Adjust as needed
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
