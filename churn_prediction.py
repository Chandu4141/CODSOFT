## Customer Churn Prediction
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

data = pd.read_csv("Churn_Modelling.csv");

le = LabelEncoder()
data['Geography'] = le.fit_transform(data['Geography'])
data['Gender'] = le.fit_transform(data['Gender'])

X = data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
print("Logistic Regression Report:")
print(classification_report(y_test, lr_preds))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, rf_preds))

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
print("Gradient Boosting Report:")
print(classification_report(y_test, gb_preds))

#Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(y='Exited', data=data, hue='Exited', palette='pastel', legend=False)
plt.title("Churn Distribution")
plt.ylabel("Exited (0 = No, 1 = Yes)")
plt.xlabel("Count")
plt.tight_layout()
plt.show()
