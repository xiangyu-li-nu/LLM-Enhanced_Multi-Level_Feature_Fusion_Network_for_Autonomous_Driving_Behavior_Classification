import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# 读取数据
df = pd.read_csv('extracted_features.csv')

# 分离特征和标签
X = df.drop(columns=['label'])
y = df['label']

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将特征重新形状以适应LSTM层 (samples, time_steps, features)
# 假设每个样本只有一个时间步，可以将特征视为时间步的特征
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))  # Dropout层，防止过拟合
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 定义早停回调
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 训练模型
history = model.fit(X_train, y_train, validation_split=0.1, epochs=200, batch_size=64, callbacks=[early_stopping], verbose=1)

# 评估模型
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 将预测结果转回类别编码
y_train_pred_classes = np.argmax(y_train_pred, axis=1)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_train_true_classes = np.argmax(y_train, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

# 生成分类报告
print('Classification Report on Training Set:\n')
print(classification_report(y_train_true_classes, y_train_pred_classes, target_names=label_encoder.classes_, digits=4))
print('-' * 80)
print('Classification Report on Test Set:\n')
print(classification_report(y_test_true_classes, y_test_pred_classes, target_names=label_encoder.classes_, digits=4))
print('-' * 80)