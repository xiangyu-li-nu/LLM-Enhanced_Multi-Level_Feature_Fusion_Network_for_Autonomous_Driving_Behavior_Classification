import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, \
    GlobalAveragePooling1D, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
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

# Assume each sample has 12 time steps, each with 3 features (total of 36 features)
# You can adjust this based on the actual feature count and time steps
time_steps = 12
features = X_scaled.shape[1] // time_steps

if X_scaled.shape[1] % time_steps != 0:
    raise ValueError(f"Feature count must be divisible by time steps. Current feature count: {X_scaled.shape[1]}, Time steps: {time_steps}")

# Reshape data for the model
X_scaled = X_scaled.reshape((X_scaled.shape[0], time_steps, features))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)


# Define Transformer Encoder Layer
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection and layer normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Residual connection and layer normalization


# Build TST model
def build_tst_model(input_shape, num_classes, embed_dim=64, num_heads=4, ff_dim=128, num_transformer_blocks=2,
                    dropout_rate=0.1):
    inputs = Input(shape=input_shape)

    # Linear transformation to embedding space
    x = Dense(embed_dim)(inputs)

    # Add multiple Transformer Encoder layers
    for _ in range(num_transformer_blocks):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, rate=dropout_rate)(x)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)

    # Fully connected layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Define model parameters
input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
num_classes = y_train.shape[1]

# Build the model
model = build_tst_model(
    input_shape=input_shape,
    num_classes=num_classes,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_transformer_blocks=2,
    dropout_rate=0.1
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
