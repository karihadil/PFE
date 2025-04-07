
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
df = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Bureau\\PFE\\backend\\app\\models\\filtered_dataset.csv')
df['url_length'] = df['url'].apply(lambda x: len(str(x)))
def count_special_chars(url):
    special_chars = "@-_%=&?"
    return sum(url.count(char) for char in special_chars)
df['special_chars_count'] = df['url'].apply(count_special_chars)
print(df['special_chars_count'].head())
df["https"] = df["url"].apply(lambda x: 1 if "https" in x else 0)
df["domain_name"] = df["url"].apply(lambda x: tldextract.extract(x).domain)
df["domain_len"]  =df["domain_name"].apply(lambda x:len(str(x)))
def check_ip(url):
    try:
        # Extract hostname from URL
        hostname = urlparse(url).hostname
        # If hostname is None (invalid URL), return 0
        if hostname is None:
            return 0
        # Check if the hostname is an IP address
        ip_address(hostname)
        return 1  # It's an IP address
    except ValueError:
        return 0  # Not an IP address
df['is_ip'] = df['url'].apply(check_ip)
print(df['is_ip'].head())
def count_subdomains(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain.strip()  # Remove extra spaces
    return len(subdomain.split(".")) if subdomain else 0  # Count subdomain parts

df["sub_nb"]=df["url"].apply(count_subdomains)
df = df[[col for col in df.columns if col != "status"] + ["status"]]
df.info()
print(df.describe())
print(df.isnull().sum())
df.dropna(inplace=True)
print(df["status"].value_counts())
print(df.head())
encoder=LabelEncoder()
df["status"]=encoder.fit_transform(df["status"]) #0 legit 1 phishing
print(df.head())
y=df["status"].value_counts()
labels = ["Legitimate", "Phishing"]
plt.figure(figsize=(10,5))
colors = ["green","red"]
plt.bar(labels,y,color=colors)
plt.title("Phishing vs Legitimate websites")
plt.show()
x1=df["https"].value_counts()
labels = ["HTTP", "HTTPS"]
plt.figure(figsize=(10,5))
colors = ["pink","purple"]
plt.barh(labels,x1,color=colors)
plt.title("HTTP vs HTTPS")
plt.show()
x2=df["is_ip"].value_counts()
labels = ["Not IP", "IP"]
plt.figure(figsize=(10,5))
colors = ["orange","blue"]
plt.bar(labels,x2,color=colors)
plt.title("IP vs Not IP")
plt.show()
phishing_keywords = ["login", "secure", "bank", "update", "verify", "account", "password"]

def contains_phishing_words(url):
    return any(word in url.lower() for word in phishing_keywords)

df["contains_phishing_words"] = df["url"].apply(contains_phishing_words).astype(int)
print(df["contains_phishing_words"].head())
df["contains_phishing_words"].value_counts()

def count_digits_in_domain(url):
    domain = urlparse(url).netloc  # Extract domain
    return sum(c.isdigit() for c in domain)
df["num_digits_in_domain"] = df["url"].apply(count_digits_in_domain)


def calculate_entropy(url):
    counter = Counter(url)  # Count occurrences of each character
    length = len(url)
    entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())
    return entropy

df["url_entropy"] = df["url"].apply(calculate_entropy)
df["dm_entropy"] = df["domain_name"].apply(calculate_entropy)

X = df.drop(columns=["status"]) # Features
y = df["status"]# Target variable

X= X.drop(columns=["domain_name", "is_ip" , "https"], errors="ignore")

categorical_cols = X.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])
print(X.dtypes)
scale=MinMaxScaler()
X_scaled=scale.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42 , stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

rf=RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})

feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
# Print top features
print(feature_importance_df)
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance_df["Importance"], y=feature_importance_df["Feature"])
plt.title("Feature Importance in Detecting Phishing URLs")
plt.show()
print(X.columns)
rf_pred=rf.predict(X_test)
accuracy_score(y_test,rf_pred)  
print(classification_report(y_test,rf_pred)) 

print("xgggggggggggbbbbbbbbbbbbbbbbbbb")
import optuna
best_params = {
    'n_estimators': 447,
    'max_depth': 10,
    'learning_rate': 0.12509164598021782,
    'min_child_weight': 1,
    'subsample': 0.9144900383287264,
    'colsample_bytree': 0.654538382686467,
    'gamma': 3.734548680958295e-08,
    'random_state': 42
}

xgb = XGBClassifier(**best_params)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

accuracy = accuracy_score(y_test, xgb_pred)
print(f"Final Accuracy with Optimized XGBoost: {accuracy:.4f}")

# Classification Report
print(classification_report(y_test, xgb_pred))
final_preds = (xgb_pred + rf_pred) / 2  # Soft voting
final_preds = np.round(final_preds).astype(int)  # Convert to binary

stacking_accuracy = accuracy_score(y_test, final_preds)
print(f"blended Model Accuracy: {stacking_accuracy:.4f}")
