import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline


def feature_extraction(data, method='count', **kwargs):
    """
    特征提取方法的参数化切换机制。

    参数：
    - data: 输入文本数据（列表或pandas的Series）。
    - method: 特征提取方法，'count' 为高频计数特征，'tfidf' 为 TF-IDF 特征。
    - kwargs: 其他参数，将传递给相应的向量化方法。

    返回：
    - 特征矩阵：提取的特征矩阵。
    """
    if method == 'count':
        vectorizer = CountVectorizer(**kwargs)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(**kwargs)
    else:
        raise ValueError("Unsupported method! Choose 'count' or 'tfidf'.")


    features = vectorizer.fit_transform(data)

    return features, vectorizer.get_feature_names_out()


# 示例数据
documents = [
    "机器学习是人工智能的一个分支",
    "深度学习是机器学习的一个重要方向",
    "自然语言处理是人工智能重要的应用之一"
]

# 使用高频词特征
count_features, count_feature_names = feature_extraction(documents, method='count')
print("高频词特征矩阵：")
print(count_features.toarray())
print("特征名：", count_feature_names)

# 使用TF-IDF加权特征
tfidf_features, tfidf_feature_names = feature_extraction(documents, method='tfidf')
print("\nTF-IDF特征矩阵：")
print(tfidf_features.toarray())
print("特征名：", tfidf_feature_names)