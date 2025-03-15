from cProfile import label
from ipaddress import ip_address
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import re
from ipaddress import ip_address
from urllib.parse import urlparse
import tldextract
import seaborn as sns
df = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Bureau\\PFE\\backend\\app\\models\\filtered_dataset.csv')
df['url_length'] = df['url'].apply(lambda x: len(str(x)))
def count_special_chars(url):
    special_chars = "@-_%=&?"
    return sum(url.count(char) for char in special_chars)
df['special_chars_count'] = df['url'].apply(count_special_chars)
print(df['special_chars_count'].head())
df["https"] = df["url"].apply(lambda x: 1 if "https" in x else 0)
df['domain_name'] = df['url'].apply(lambda x: urlparse(x).netloc)
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
X = df.drop(columns=["status"]) # Features
y = df["status"]# Target variable

from collections import Counter
import math

def calculate_entropy(url):
    counter = Counter(url)  # Count occurrences of each character
    length = len(url)
    entropy = -sum((count/length) * math.log2(count/length) for count in counter.values())
    return entropy

df["url_entropy"] = df["url"].apply(calculate_entropy)

X= X.drop(columns=["url", "domain_name"], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 , stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
from sklearn.preprocessing import LabelEncoder
categorical_cols = X.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])
print(X.dtypes)

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