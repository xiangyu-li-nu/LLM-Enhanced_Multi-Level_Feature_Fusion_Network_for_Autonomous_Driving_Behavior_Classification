import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# Read data
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

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)

# Reshape input for Conv1D
# Conv1D expects input shape (samples, timesteps, features)
# Here, we treat each feature of a sample as a timestep with 1 feature per timestep
X_train_conv = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_conv = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the FCN model
model = Sequential()

# First convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_conv.shape[1], 1)))
model.add(Dropout(0.5))

# Second convolutional layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))

# Global average pooling layer
model.add(GlobalAveragePooling1D())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_conv, y_train,
    validation_split=0.1,
    epochs=50,  # Increase epochs to fully train
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
y_train_pred = model.predict(X_train_conv)
y_test_pred = model.predict(X_test_conv)

# Convert prediction results back to class labels
y_train_pred_classes = np.argmax(y_train_pred, axis=1)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_train_true_classes = np.argmax(y_train, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

# Generate classification report
print('Classification Report on Training Set:\n')
print(classification_report(
    y_train_true_classes,
    y_train_pred_classes,
    target_names=label_encoder.classes_,
    digits=4
))
print('-' * 80)
print('Classification Report on Test Set:\n')
print(classification_report(
    y_test_true_classes,
    y_test_pred_classes,
    target_names=label_encoder.classes_,
    digits=4
))
print('-' * 80)
