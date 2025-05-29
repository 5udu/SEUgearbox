import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pylab import mpl
from hho import HarrisHawkOptimization

# 设置中文显示
mpl.rcParams['font.sans-serif'] = ['STZhongsong']
mpl.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = "features_reconstructed_simplified_df.csv"
df = pd.read_csv(file_path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
selected_features = ["整流平均值", "均方根", "低频能量", "低频奇异值特征", "高频能量", "频带能量", "平均值", "均方频率", "重心频率"]
X = df[selected_features]
y = df["故障类别"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def fitness_function(params):
    hidden1, hidden2, learning_rate = int(params[0]), int(params[1]), params[2]
    hidden1 = max(10, min(hidden1, 200))  # 限制范围
    hidden2 = max(5, min(hidden2, 200))
    learning_rate = round(max(0.0001, min(learning_rate, 0.1)), 9)

    clf = MLPClassifier(
        hidden_layer_sizes=(hidden1, hidden2),
        activation='relu',
        solver='adam',
        learning_rate_init=learning_rate,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    return -acc, (hidden1, hidden2, learning_rate)  # 负数用于最小化目标

# 设置搜索边界
lb = [50, 50, 0.001]
ub = [200, 200, 0.1]

# 初始化 HHO
hho = HarrisHawkOptimization(
    fitness_function=fitness_function,
    n_hawks=20,
    dim=3,
    max_iter=100,
    lb=lb,
    ub=ub
)

# 执行优化
_, best_accuracy, best_params = hho.optimize()

hidden1, hidden2, learning_rate = best_params

final_model = MLPClassifier(
    hidden_layer_sizes=(hidden1, hidden2),
    activation='relu',
    solver='adam',
    learning_rate_init=learning_rate,
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n=== 优化完成 ===")
print(f"最佳参数 - 隐藏层1: {hidden1}, 隐藏层2: {hidden2}, 学习率: {learning_rate}")
print(f"最终准确率: {accuracy:.4f}")
print("分类报告:")
print(report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=final_model.classes_,
            yticklabels=final_model.classes_)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("HHO-BP 神经网络 - 故障分类混淆矩阵（优化后）")
plt.show()