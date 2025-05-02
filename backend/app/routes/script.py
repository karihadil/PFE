import joblib
import re
from urllib.parse import urlparse
import tldextract
import math
from collections import Counter
import numpy as np
import pandas as pd

# -----------------------------
# Load the saved model and features
# -----------------------------
xgb_model = joblib.load('phishing_detector_xgb.pkl')
features = joblib.load('features.pkl')

# -----------------------------
# Feature extraction functions
# -----------------------------
def abnormal_url(url):
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"
    return 1 if domain not in url else 0

def dot_count_hostname(url):
    hostname = urlparse(url).hostname
    return hostname.count('.') if hostname else 0

def count_special_chars(url):
    special_chars = "@-_%=&?"
    return sum(url.count(char) for char in special_chars)

def no_of_dir(url):
    return urlparse(url).path.count('/')

def shortening_service(url):
    domain = urlparse(url).netloc
    match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                    r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                    r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                    r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                    r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                    r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                    r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                    r'tr\.im|link\.zip\.net', domain)
    return 1 if match else 0

def get_hostname_length(url):
    hostname = urlparse(url).hostname
    return len(hostname) if hostname else 0

def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def extract_tld(url):
    extracted = tldextract.extract(url)
    return extracted.suffix

def digit_count(url):
    return sum(c.isdigit() for c in url)

def letter_count(url):
    return sum(c.isalpha() for c in url)

def calculate_entropy(url):
    char_counts = Counter(url)
    total_chars = len(url)
    entropy = -sum((count / total_chars) * math.log2(count / total_chars) for count in char_counts.values())
    return entropy

# -----------------------------
# Function to extract all features from URL
# -----------------------------
def extract_features(url):
    return {
        'url_len': len(url),
        'abnormal_url': abnormal_url(url),
        'count_dot_hostname': dot_count_hostname(url),
        'count-www': url.count('www'),
        'count@': url.count('@'),
        'special_chars_count': count_special_chars(url),
        'https': 1 if 'https' in url else 0,
        'domain_len': len(tldextract.extract(url).domain),
        'count_dir': no_of_dir(url),
        'short_url': shortening_service(url),
        'count-https': url.count('https'),
        'count-http': url.count('http'),
        'count%': url.count('%'),
        'count-': url.count('-'),
        'count=': url.count('='),
        'hostname_len': get_hostname_length(url),
        'fd_length': fd_length(url),
        'tld_len': len(str(extract_tld(url))),
        'count-digits': digit_count(url),
        'count-letters': letter_count(url),
    }




def predict_url(url):
    features_dict = extract_features(url)
    features_df = pd.DataFrame([features_dict])
    prediction = xgb_model.predict(features_df)
    prediction_label = 'Phishing' if prediction[0] == 1 else 'Legitimate'
    return prediction_label

if __name__ == "__main__":
    url = input("Enter the URL to check: ").strip()
    if not url:
        print("❗ No URL entered. Please provide a valid URL.")
    else:
        result = predict_url(url)
        print(f"\n✅ The URL is: {result}")
