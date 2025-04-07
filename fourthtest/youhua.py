import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

data = [
    "This is a spam email",
    "Hello, I want to talk about your project",
    "Congratulations, you have won a lottery",
    "Important, your account has been compromised",
    "Just checking in to see how things are going",
    "Your bank account has been compromised",
    "Don't miss this limited time offer"
]

labels = [1, 0, 1, 1, 0, 1, 1]

df = pd.DataFrame({'text': data, 'label': labels})

def extract_features(X, method='tfidf'):
    """
    根据选择的特征提取方法返回特征矩阵。

    参数：
    - X: 输入文本数据（列表或Series）
    - method: 'tfidf'，表示使用TF-IDF特征提取

    返回：
    - 特征矩阵
    """
    if method == 'tfidf':
        # 使用TF-IDF加权特征
        vectorizer = TfidfVectorizer(max_features=10)  # 取前10个特征
        X_features = vectorizer.fit_transform(X)

    else:
        raise ValueError("Invalid method. Currently only 'tfidf' is supported.")

    return X_features


# 使用extract_features函数来提取特征
X = df['text']
y = df['label']

# 选择特征提取方法，可以切换为 'tfidf'
method = 'tfidf'  # 目前仅支持'tfidf'
X_features = extract_features(X, method=method)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)

# 使用随机森林进行训练
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 预测并评估模型
y_pred = model.predict(X_test)

# 打印分类报告和准确率
print("分类报告:\n", classification_report(y_test, y_pred))
print("准确率: ", accuracy_score(y_test, y_pred))