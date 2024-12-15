import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D, concatenate
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

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

# Reshape the features to fit LSTM and Conv1D layers (samples, time_steps, features)
# Assume each sample has only one time step, so features can be treated as the features for the time step
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)

# Build the LSTM-FCN model
# Input layer
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# LSTM branch
lstm_out = LSTM(128, return_sequences=True)(input_layer)
lstm_out = Dropout(0.5)(lstm_out)
lstm_out = LSTM(64, return_sequences=False)(lstm_out)
lstm_out = Dropout(0.5)(lstm_out)

# FCN branch
conv_out = Conv1D(filters=128, kernel_size=1, activation='relu')(input_layer)
conv_out = Dropout(0.5)(conv_out)
conv_out = Conv1D(filters=256, kernel_size=1, activation='relu')(conv_out)
conv_out = Dropout(0.5)(conv_out)
conv_out = Conv1D(filters=128, kernel_size=1, activation='relu')(conv_out)
conv_out = GlobalAveragePooling1D()(conv_out)

# Combine the LSTM and FCN branches
combined = concatenate([lstm_out, conv_out])

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

# Print the model structure
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
    epochs=20,  # Increase epochs for sufficient training
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert the predictions back to class labels
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
