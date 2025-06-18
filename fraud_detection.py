# **** Fraud Detection ****
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load datasets
train_data = pd.read_csv("fraudTrain.csv").sample(frac=0.2, random_state=42)
test_data = pd.read_csv("fraudTest.csv")

# Check class distribution
print("Training set fraud count:\n", train_data['is_fraud'].value_counts())

# Preprocessing
def preprocess(df):
    df = df.copy()
    drop_cols = ['trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city',
                 'state', 'zip', 'dob', 'trans_num']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.fillna('missing', inplace=True)
    for col in ['category', 'gender', 'job', 'merchant']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    numeric_cols = ['amt', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
train_data = preprocess(train_data)
test_data = preprocess(test_data)
X_train = train_data.drop(columns=['is_fraud'])
y_train = train_data['is_fraud']
X_test = test_data.drop(columns=['is_fraud'])
y_test = test_data['is_fraud']

# Initialize models
log_model = LogisticRegression(max_iter=1000, solver='liblinear')
dt_model = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
rf_model = RandomForestClassifier(n_estimators=20, max_depth=10, class_weight='balanced', random_state=42)

# Train and predict: Logistic Regression
log_model.fit(X_train, y_train)
y_pred_log_reg = log_model.predict(X_test)

# Train and predict: Decision Tree
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Train and predict: Random Forest
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# **** Model Evaluation ****
# Logistic Regression
print("\n======  Logistic Regression:  ======")
log_acc = accuracy_score(y_test, y_pred_log_reg)
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log_reg, zero_division=0))

# Decision Tree
print("=====  Decision Tree:  =====")
dt_acc = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt, zero_division=0))

# Random Forest
print("=====  Random Forest:  =====")
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))

# Accuracy Plot
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [log_acc, dt_acc, rf_acc]
plt.figure(figsize=(6, 5))
bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.title("Model Accuracy Comparison",pad=15)
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.4f}', ha='center', fontsize=10)
plt.tight_layout()
plt.show()