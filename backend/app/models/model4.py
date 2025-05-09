
import pandas as pd
import re
from urllib.parse import urlparse
import tldextract
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tld import get_tld
import os.path
import numpy as np
import math
from collections import Counter

df = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Bureau\\PFE\\backend\\app\\models\\filtered_dataset.csv')
print(df.head()) 
print(df.isna().sum())
print(df['status'].value_counts())
df['url_len'] = [len(url) for url in df.url]
print(df['url_len'].head())
#the distribution of url length depending on the type of url (legit or phishing)
sns.displot(df, x='url_len', hue='status', kind='kde', fill=True)
plt.xlabel('URL Length')
plt.ylabel('Density')
plt.title('Distribution of URL Length by Type')
plt.show()
encoder=LabelEncoder()
# detects if the url contains an ip address
def abnormal_url(url):
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"
    if domain not in url:
        return 1  # Abnormal
    else:
        return 0  # Normal
df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))
def dot_count_hostname(url):
    hostname = urlparse(url).hostname
    return hostname.count('.') if hostname else 0
df["count_dot_hostname"] = df["url"].apply(lambda x: dot_count_hostname(x))
df['count-www'] = df['url'].apply(lambda i: i.count('www'))
df["count@"]    = df["url"].apply(lambda x: x.count("@"))
def count_special_chars(url):
    special_chars = "@-_%=&?"
    return sum(url.count(char) for char in special_chars)
df['special_chars_count'] = df['url'].apply(count_special_chars)
df["https"] = df["url"].apply(lambda x: 1 if "https" in x else 0)
df["domain_name"] = df["url"].apply(lambda x: tldextract.extract(x).domain)
df["domain_len"]  =df["domain_name"].apply(lambda x:len(str(x)))
#Count Dir / URL Depth¶
def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))
def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')
def shortening_service(url):
    domain = urlparse(url).netloc
    match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                    r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                    r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                    r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                    r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                    r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                    r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                    r'tr\.im|link\.zip\.net',
                    domain)
    if match:
        return 1
    else:
        return 0

df['short_url'] = df['url'].apply(lambda i: shortening_service(i))
df['count-https'] = df['url'].apply(lambda i : i.count('https'))
df['count-http'] = df['url'].apply(lambda i : i.count('http'))
df['count%'] = df['url'].apply(lambda i: i.count('%'))
df['count-'] = df['url'].apply(lambda i: i.count('-'))
df['count='] = df['url'].apply(lambda i: i.count('='))
def get_hostname_length(url):
    try:
        hostname = urlparse(url).hostname
        return len(hostname) if hostname else 0
    except:
        return 0

df['hostname_len'] = df['url'].apply(get_hostname_length)
#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

df['fd_length'] = df['url'].apply(lambda i: fd_length(i))
#tld length
def extract_tld(url):
    extracted = tldextract.extract(url) 
    return extracted.suffix  # This gives you the TLD 
df["tld_len"]= df["url"].apply(lambda x: len(str(extract_tld(x)))) 
def has_suspicious_words(url):
    keywords = ['login', 'secure', 'account', 'update', 'free', 'verify', 'password', 'ebayisapi', 'banking', 'signin']
    return int(any(word in url.lower() for word in keywords))
df['suspicious_words'] = df['url'].apply(lambda i: has_suspicious_words(i))
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

df['count-digits']= df['url'].apply(lambda i: digit_count(i))
def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

df['count-letters']= df['url'].apply(lambda i: letter_count(i))

def calculate_entropy(url):
    # Count frequency of each character
    char_counts = Counter(url)
    total_chars = len(url)
    
    # Shannon entropy formula
    entropy = -sum((count / total_chars) * math.log2(count / total_chars) for count in char_counts.values())
    return entropy
df['url_entropy'] = df['url'].apply(calculate_entropy)
def url_path_length(url):
    return len(urlparse(url).path)
df['url_path_length'] = df['url'].apply(url_path_length)
TRUSTED_DOMAINS = {'google.com',
    'github.com',
    'wikipedia.org',
    'apple.com',
    'linkedin.com',
    'microsoft.com',
    'facebook.com',
    'amazon.com',
    'paypal.com',
    'dropbox.com',
    'youtube.com',
    'openai.com',
    'mozilla.org',
    'cloudflare.com',
    'netflix.com',
    'office.com',
    'whatsapp.com',
    'zoom.us',
    'adobe.com',
    'stackoverflow.com'}

def is_trusted_domain(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    return any(hostname.endswith(td) for td in TRUSTED_DOMAINS)
df['trusted_domain'] = df['url'].apply(is_trusted_domain)
def subdomain_count(url):
    hostname = urlparse(url).hostname or ''
    return hostname.count('.') - 1
df['subdomain_count'] = df['url'].apply(subdomain_count)

def move_status_to_end(df):
    cols = list(df.columns)
    if 'status' in cols:
        cols.append(cols.pop(cols.index('status')))  # Remove 'status' and add it to the end
        return df[cols]
    return df  # Return unchanged if 'status' is not found
df= move_status_to_end(df)
encoder=LabelEncoder()
df["status"]=encoder.fit_transform(df["status"]) #0 legit 1 phishing
df.dropna(inplace=True)
df.drop_duplicates()
print(df.describe())
print(df.isnull().sum())
print(df.head())
print(df.tail())
df.to_csv('C:\\Users\\DELL\\OneDrive\\Bureau\\PFE\\backend\\app\\models\\processed_dataset.csv', index=False)
print(df.columns.tolist())
X = df[['url_len', 'abnormal_url', 'count_dot_hostname', 'count@',
                'special_chars_count', 'https', 'domain_len', 'count_dir', 'short_url',
                'count-https', 'count-http', 'count%', 'count-', 'count=',
                'hostname_len', 'fd_length', 'tld_len', 'count-digits', 'count-letters','trusted_domain','subdomain_count','suspicious_words','url_path_length','url_entropy']]

y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()

scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validated accuracy: {scores.mean()*100:.2f}%")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracies
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
train_sizes, train_scores, test_scores = learning_curve( # type: ignore
    model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy', marker='o')
plt.plot(train_sizes, test_scores_mean, label='Validation Accuracy', marker='s')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 
print('xgggggggggggggggggggggggggggggggggggggggggggggggggggggggggbooooooooooooooossssssssssssttttttttt')
import warnings
warnings.filterwarnings("ignore")
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))
print(f"Accuracy: {accuracy * 100:.2f}%")

train_sizes, train_scores, test_scores = learning_curve( # type: ignore
    xgb_model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42
)

# Accuracy of trainset and testset
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Print accuracies
print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy', marker='o')
plt.plot(train_sizes, test_scores_mean, label='Validation Accuracy', marker='s')
plt.title('Learning Curve - XGBoost Model')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validated accuracy: {xgb_cv_scores.mean() * 100:.2f}%")

import joblib

joblib.dump(xgb_model, 'phishing_detector_xgb.pkl')
features = ['url_len', 'abnormal_url', 'count_dot_hostname', 'count@',
                'special_chars_count', 'https', 'domain_len', 'count_dir', 'short_url',
                'count-https', 'count-http', 'count%', 'count-', 'count=',
                'hostname_len', 'fd_length', 'tld_len', 'count-digits', 'count-letters','trusted_domain','subdomain_count','suspicious_words','url_path_length','url_entropy']

joblib.dump(features, 'features.pkl')
print("fusion model")
fusion_model = VotingClassifier(estimators=[
    ('rf', model),
    ('xgb', xgb_model)
], voting='soft', n_jobs=-1)

# Train on SMOTE-resampled training set
fusion_model.fit(X_train, y_train)

# Predict on test set
y_test_pred = fusion_model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_test_pred)
print(f"\n✅ Fusion Model Accuracy: {accuracy * 100:.2f}%")

print("\n🧾 Classification Report (Fusion Model):")
print(classification_report(y_test, y_test_pred))

# Cross-validation
from sklearn.model_selection import cross_val_score
fusion_cv_scores = cross_val_score(fusion_model, X, y, cv=5, scoring='accuracy')
print(f"📊 Cross-validated accuracy (Fusion Model): {fusion_cv_scores.mean() * 100:.2f}%")
import joblib
joblib.dump(fusion_model, 'fusion_model_final.pkl')

