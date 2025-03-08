from ipaddress import ip_address
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import re
from ipaddress import ip_address
from urllib.parse import urlparse
import tldextract
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
print(df.head())