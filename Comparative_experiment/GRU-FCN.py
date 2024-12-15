import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from keras.models import Model
from keras.layers import Input, GRU, Dense, Dropout, Conv1D, GlobalAveragePooling1D, concatenate
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('extracted_features.csv')

# Separate features and labels
X = df.drop(columns=['label'])
y = df['label']

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Assume each sample has 6 time steps and 6 features per time step
# 36 = 6 * 6
time_steps = 6
features = X_scaled.shape[1] // time_steps
if X_scaled.shape[1] % time_steps != 0:
    raise ValueError(f"Number of features must be divisible by the number of time steps. Current feature count: {X_scaled.shape[1]}, time steps: {time_steps}")

X_scaled = X_scaled.reshape((X_scaled.shape[0], time_steps, features))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)

# Build GRU-FCN model
# Input layer
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# GRU branch
gru_out = GRU(128, return_sequences=True)(input_layer)
gru_out = Dropout(0.5)(gru_out)
gru_out = GRU(64, return_sequences=False)(gru_out)
gru_out = Dropout(0.5)(gru_out)

# FCN (Fully Convolutional Network) branch
conv_out = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(input_layer)
conv_out = Dropout(0.5)(conv_out)
conv_out = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(conv_out)
conv_out = Dropout(0.5)(conv_out)
conv_out = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv_out)
conv_out = GlobalAveragePooling1D()(conv_out)

# Merge GRU and FCN branches
combined = concatenate([gru_out, conv_out])

# Fully connected layer
dense_out = Dense(128, activation='relu')(combined)
dense_out = Dropout(0.5)(dense_out)

# Output layer
output_layer = Dense(y_train.shape[1], activation='softmax')(dense_out)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

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
    epochs=20,  # Increase epochs to fully train
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