import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ------------------- Load Dataset -------------------
data = pd.read_csv("./Dataset/scurvy.csv")

# Encode ordered categorical columns
ordered_cols = ['gum_rot_d6', 'skin_sores_d6', 'weakness_of_the_knees_d6', 'lassitude_d6']
le = LabelEncoder()
for col in ordered_cols:
    data[col] = le.fit_transform(data[col])

# One-hot encode treatment and convert to int
data = pd.get_dummies(data, columns=['treatment'])
for col in data.columns:
    if col.startswith("treatment_"):
        data[col] = data[col].astype(int)

# Encode target
data['fit_for_duty_d6'] = data['fit_for_duty_d6'].map({'0_no': 0, '1_yes': 1})

# Features and target
X = data.drop(['study_id', 'dosing_regimen_for_scurvy', 'fit_for_duty_d6'], axis=1)
y = data['fit_for_duty_d6']

print("Original class distribution:\n", y.value_counts())

# ------------------- Safe Synthetic Augmentation -------------------
minority_idx = np.where(y==1)[0]
minority_sample = X.iloc[minority_idx]

# Generate 10 synthetic samples with small noise
augmented_samples = []
augmented_targets = []
for i in range(10):
    noise = np.random.normal(0, 0.01, size=minority_sample.shape)
    new_sample = minority_sample + noise
    augmented_samples.append(new_sample.values[0])
    augmented_targets.append(1)

X_aug = pd.concat([X, pd.DataFrame(augmented_samples, columns=X.columns)], ignore_index=True)
y_aug = pd.concat([y, pd.Series(augmented_targets)], ignore_index=True)

print("\nAfter augmentation class distribution:\n", y_aug.value_counts())

# ------------------- Train/Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.3, random_state=42, stratify=y_aug
)

# ------------------- Train Random Forest -------------------
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# ------------------- Evaluation -------------------
print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# ------------------- Feature Importance -------------------
feat_importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Random Forest Feature Importance")
plt.show()

# ------------------- SHAP Explainability -------------------
explainer_rf = shap.Explainer(rf_clf, X_train)
shap_values_rf = explainer_rf(X_test)
shap.summary_plot(shap_values_rf, X_test, plot_type="bar", show=True)
