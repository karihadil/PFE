import numpy as np
import pandas as pd
import math
from collections import Counter
import re
from urllib.parse import urlparse
import tldextract
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ipaddress import ip_address
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('C:\\Users\\DELL\\OneDrive\\Bureau\\PFE\\backend\\app\\models\\dataset_phishing.csv')
print(df.head())
print(df.info())
print(df.describe())
x=df.drop(columns=["status"])
encoder=LabelEncoder()
df["status"]=encoder.fit_transform(df["status"])
df["status"].value_counts()
plt.figure(figsize=(8, 6))
sns.countplot(x="status", data=df)
plt.title("Distribution of Phishing and Legitimate URLs")
plt.xlabel("Status")
plt.ylabel("Count")
plt.xticks([0, 1], ["Legitimate", "Phishing"])
plt.show()
y=df["status"]
categorical_cols = x.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in categorical_cols:
    x[col] = encoder.fit_transform(x[col])
print(x.dtypes)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train,y_train)
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": x.columns, "Importance": importances})

feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
top_10_features = feature_importance_df.head(15)
print("\nTop 10 Important Features:")
print(top_10_features)

# Plot top 10 features
plt.figure(figsize=(10, 5))
sns.barplot(x=top_10_features["Importance"], y=top_10_features["Feature"])
plt.title("Top 10 Feature Importances in Detecting Phishing URLs")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
selected_columns = [
    "google_index",
    "page_rank",
    "nb_hyperlinks",
    "web_traffic",
    "nb_www",
    "longest_word_path",
    "domain_age",
    "ratio_intHyperlinks",
    "ratio_extHyperlinks",
    "phish_hints",
    "safe_anchor",
    "ratio_digits_url",
    "ratio_extRedirection",
    "avg_word_path"
    
]
filtered_df = df[selected_columns + ["status"]].dropna().drop_duplicates().reset_index(drop=True)
filtered_df.to_csv("C:\\Users\\DELL\\OneDrive\\Bureau\\PFE\\backend\\app\\models\\dataset_phishing_final.csv", index=False)
x = filtered_df[selected_columns]
y = filtered_df["status"]
print(x.info())
print(x.describe())
print(x.head())
print(y.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100, random_state=42,max_depth=10)
model.fit(x_train,y_train)
rf_pred=model.predict(x_test)
print(classification_report(y_test,rf_pred))
print(confusion_matrix(y_test,rf_pred))
print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
from sklearn.metrics import accuracy_score


y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
mean_train_accuracy = cv_scores.mean()


model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Mean Train Accuracy (CV): {mean_train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
plt.figure(figsize=(6, 4))
plt.bar(["Train (CV Mean)", "Test"], np.array([mean_train_accuracy, test_accuracy]), color=["steelblue", "indianred"])
plt.title("Cross-Validated Train Accuracy vs Test Accuracy")
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
for i, acc in enumerate([mean_train_accuracy, test_accuracy]):
    plt.text(i, float(acc) + 0.02, f"{float(acc):.2f}", ha='center', fontsize=12)
plt.tight_layout()
plt.show()
xgb_model=XGBClassifier(n_estimators=100, random_state=42, max_depth=10)
xgb_model.fit(x_train, y_train)
xgb_pred=xgb_model.predict(x_test)
print("XGBoost Classifier")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Classification Report:\n", classification_report(y_test, xgb_pred))
import joblib
# Save the model
joblib.dump(model, "phishing_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
