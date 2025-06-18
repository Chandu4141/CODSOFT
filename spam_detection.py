# Spam Detection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("spam.csv",encoding='latin-1')
print("Last few rows of the dataset:\n", df.tail())
print("\nColumn are: ", df.columns)

df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df = df[['label', 'message']]

df.isnull().sum()
df.dropna(inplace=True)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

log_model = LogisticRegression()
nb_model = MultinomialNB()
svm_model = SVC()
results = {}

# Logistic Regression
log_model.fit(X_train, y_train)
y_pred_lr = log_model.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
results['Logistic_Reg'] = acc_lr
print("\n**** Logistic Regression ****")
print(f"Accuracy: {acc_lr:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# Naive Bayes
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
results['Naive_Bayes'] = acc_nb
print("\n**** Naive Bayes ****")
print(f"Accuracy: {acc_nb:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

# Support Vector Machine
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
results['Support_Vector_Machine'] = acc_svm
print("\n**** Support Vector Machine ****")
print(f"Accuracy: {acc_svm:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

# Plotting
plt.figure(figsize=(6, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Spam Classification Accuracy by Model")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1)
plt.tight_layout()
plt.show()
