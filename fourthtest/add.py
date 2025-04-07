import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

spam_count = 127
ham_count = 24

data = {
    'text': ['This is spam email {}'.format(i) for i in range(1, spam_count + 1)] +
            ['This is normal email {}'.format(i) for i in range(1, ham_count + 1)],
    'label': ['spam'] * spam_count + ['ham'] * ham_count
}

df = pd.DataFrame(data)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_vectorized, y_train)

model = MultinomialNB()
model.fit(X_resampled, y_resampled)

X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)

print(classification_report(y_test, y_pred))