import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Multiply, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# Define gMLP Block
class gMLPBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ffn, seq_len, dropout_rate=0.1):
        super(gMLPBlock, self).__init__()
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.channel_proj1 = Dense(d_ffn, activation='gelu')  # MLP
        self.channel_proj2 = Dense(d_model)  # Projection back
        self.spatial_proj = Dense(d_ffn, activation='sigmoid')  # Spatial projection, used for gating
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=False):
        # LayerNorm
        y = self.layernorm(x)

        # Channel Projection (MLP)
        y = self.channel_proj1(y)

        # Spatial Projection (gating)
        s = self.spatial_proj(x)  # shape: (batch_size, seq_len, d_ffn)

        # Ensure s has the same shape as y for multiplication
        # If d_ffn != d_model, adjust the spatial_proj output accordingly
        # Here, assuming d_ffn == d_model for simplicity
        y = Multiply()([y, s])  # shape: (batch_size, seq_len, d_ffn)

        # Channel Projection Back
        y = self.channel_proj2(y)  # shape: (batch_size, seq_len, d_model)

        # Dropout
        y = self.dropout(y, training=training)

        # Residual Connection
        return x + y


# Build gMLP Model
def build_gmlp_model(input_shape, num_classes, d_model=128, d_ffn=128, seq_len=12, dropout_rate=0.1, num_blocks=4):
    inputs = Input(shape=input_shape)

    # Initial linear projection
    x = Dense(d_model)(inputs)  # shape: (batch_size, seq_len, d_model)

    # Add multiple gMLP Blocks
    for _ in range(num_blocks):
        x = gMLPBlock(d_model, d_ffn, seq_len, dropout_rate)(x)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)

    # Fully connected layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Read data
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

# Assume each sample has 12 time steps, and each time step has 3 features (total features = 36)
# You can adjust this according to the actual number of features and time steps
time_steps = 12
features = X_scaled.shape[1] // time_steps

if X_scaled.shape[1] % time_steps != 0:
    raise ValueError(f"Feature count must be divisible by the time steps. Current feature count: {X_scaled.shape[1]}, time steps: {time_steps}")

X_scaled = X_scaled.reshape((X_scaled.shape[0], time_steps, features))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Define model parameters
input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, features)
num_classes = y_train.shape[1]

# Build the model
model = build_gmlp_model(
    input_shape=input_shape,
    num_classes=num_classes,
    d_model=128,
    d_ffn=128,  # Ensure d_ffn == d_model for compatibility
    seq_len=time_steps,
    dropout_rate=0.1,
    num_blocks=4
)

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
    epochs=10,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert prediction results back to class labels
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
