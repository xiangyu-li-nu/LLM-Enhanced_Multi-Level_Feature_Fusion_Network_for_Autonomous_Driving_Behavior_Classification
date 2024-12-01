import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# 读取数据
df = pd.read_csv('extracted_features.csv')

# 分离特征和标签
X = df.drop(columns=['label'])
y = df['label']

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 定义模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naive Bayes': GaussianNB(),
    'Multilayer Perceptron': MLPClassifier(max_iter=10000)
}

# 训练模型并生成分类报告
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f'Classification Report for {name} on Training Set:\n')
    print(classification_report(y_train, y_train_pred, target_names=label_encoder.classes_, digits=4))
    print('-' * 80)

    print(f'Classification Report for {name} on Test Set:\n')
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_, digits=4))
    print('-' * 80)