import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Load the dataset
csv_path = "perfectly_separable_parking.csv"
df = pd.read_csv(csv_path)
print("\U0001F4E6 Data Preview:\n", df.head())

# Encode categorical columns if any
le = LabelEncoder()
if 'Sensor_ID' in df.columns:
    df['Sensor_ID'] = le.fit_transform(df['Sensor_ID'])

# Features and target
X = df.drop('Status', axis=1)
y = df['Status']

# Encode target if needed
if y.dtype == 'O':
    y = le.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Random Forest Classifier ---
rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\u2705 Random Forest Accuracy: {acc_rf*100:.2f}%")
print("\n📊 Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# --- SVM Classifier ---
svm_model = SVC(kernel='rbf', C=0.1, gamma=2.0, probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"\U0001F3AF SVM Accuracy: {acc_svm*100:.2f}%")
print("\n📊 SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Oranges')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Random Forest metrics
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# SVM metrics
acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm)
rec_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

# Create a summary DataFrame
metrics_df = pd.DataFrame({
    'Performance Metrics': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Random Forest': [f"{acc_rf:.3f}", f"{prec_rf:.3f}", f"{rec_rf:.3f}", f"{f1_rf:.3f}"],
    'SVM': [f"{acc_svm:.3f}", f"{prec_svm:.3f}", f"{rec_svm:.3f}", f"{f1_svm:.3f}"]
})

print("\nPerformance Comparison Table:")
print(metrics_df.to_string(index=False))

# Improved bar graph for performance metrics comparison
metrics_plot = metrics_df.set_index('Performance Metrics').astype(float)
fig, ax = plt.subplots(figsize=(8, 5))
metrics_plot.plot(
    kind='bar',
    width=0.8,
    edgecolor='black',
    ax=ax
)
ax.set_title('Performance Metrics Comparison', fontsize=18)
ax.set_ylabel('Scores', fontsize=14)
ax.set_xlabel('Metrics', fontsize=14)
ax.set_ylim(0.0, 1.05)
ax.set_xticklabels(metrics_plot.index, rotation=0, fontsize=12)
ax.legend(['Random Forest', 'SVM'], fontsize=12, loc='best', frameon=True)
plt.tight_layout()
plt.show()

# Save the performance comparison table to CSV
metrics_df.to_csv('rf_svm_performance_comparison.csv', index=False)

# --- ROC-AUC Curve for Random Forest and SVM (Combined Plot) ---
from sklearn.metrics import roc_curve, auc

rf_probs = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Ensure SVM is fitted with probability=True
if not hasattr(svm_model, "predict_proba"):
    svm_model = SVC(kernel='rbf', C=0.1, gamma=2.0, probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

svm_probs = svm_model.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Combined ROC Curve Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=3, marker='o', markevery=0.1, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, color='orange', lw=3, marker='s', markevery=0.1, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest vs SVM')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve_comparison.png')
plt.show()

# Save classification reports to text files
with open('rf_classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred_rf))

with open('svm_classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred_svm))

# Plot for Random Forest
plt.figure(figsize=(12, 4))
plt.plot(y_test[:100], label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(y_pred_rf[:100], label='Predicted (RF)', marker='x', linestyle='--', alpha=0.7)
plt.title('Random Forest: Actual vs Predicted Parking Availability (Status)')
plt.xlabel('Sample Index')
plt.ylabel('Parking Status (0=Free, 1=Occupied)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot for SVM
plt.figure(figsize=(12, 4))
plt.plot(y_test[:100], label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(y_pred_svm[:100], label='Predicted (SVM)', marker='x', linestyle='--', alpha=0.7)
plt.title('SVM: Actual vs Predicted Parking Availability (Status)')
plt.xlabel('Sample Index')
plt.ylabel('Parking Status (0=Free, 1=Occupied)')
plt.legend()
plt.tight_layout()
plt.show() 