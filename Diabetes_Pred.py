# Diabetes prediction using patient health features
# Model = LogisticRegession

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    recall_score,
    roc_auc_score)

# 1. Load dataset
df = pd.read_csv("diabetes2.csv")

# print(df["Outcome"].value_counts(normalize=True))

# 2. Check missing values
# print(df.head())
# print(df.isnull().sum())

# 3. Data Cleaning
cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].median())


# 4. Feature & Target Split

X= df.drop("Outcome", axis=1)
y= df["Outcome"]


# 5. Split the data for training and testing
X_train, X_test, y_train, y_test= train_test_split(
    X,y, random_state=42, test_size=0.2, stratify=y
)

# 6. Scaling thr data
scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

# 7. Model Training
model = LogisticRegression(class_weight="balanced",max_iter=1000)
model.fit(X_train,y_train)

# 8. Predictions
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.4).astype(int)



# 9. Evaluation
print(f"Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}%")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
print("Recall: ", recall_score(y_test,y_pred))
print("Classification Report: \n", classification_report(y_test,y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test,y_pred))

# 10. Feature Importance (Logistic Regression Coefficients)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance:\n")
print(feature_importance)

# 11. Confusion Matrix Visualization
plt.figure(figsize=(6,4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
