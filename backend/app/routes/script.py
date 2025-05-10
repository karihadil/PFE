import joblib
import re
import requests
from urllib.parse import urlparse
import tldextract
import math
from collections import Counter
import numpy as np
import pandas as pd
import csv

# ====== Google Safe Browsing Setup ======
GOOGLE_API_KEY = "AIzaSyBlIx-aWXKXyQ-tAGHsEOwtZexmb5AufTs"  # Don't expose this in production

def check_google_safe_browsing(url):
    endpoint = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
    body = {
        "client": {
            "clientId": "phishing-detector",
            "clientVersion": "1.0"
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    params = {"key": GOOGLE_API_KEY}
    response = requests.post(endpoint, params=params, json=body)
    result = response.json()
    return bool(result.get("matches"))

# ====== Load Models ======
xgb_model = joblib.load('phishing_detector_xgb.pkl')
fusion_model = joblib.load('fusion_model_final.pkl')
features = joblib.load('features.pkl')

# ====== Feature Extraction ======
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
                    r'tr\.im|link\.zip\.net',
                    domain)
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
    return -sum((count / total_chars) * math.log2(count / total_chars) for count in char_counts.values())

TRUSTED_DOMAINS = {
    'google.com', 'github.com', 'wikipedia.org', 'apple.com', 'linkedin.com', 'microsoft.com',
    'facebook.com', 'amazon.com', 'paypal.com', 'dropbox.com', 'youtube.com', 'openai.com',
    'mozilla.org', 'cloudflare.com', 'netflix.com', 'office.com', 'whatsapp.com',
    'zoom.us', 'adobe.com', 'stackoverflow.com','icloud.com',
'apple.com', 'yahoo.com', 'twitter.com', 'instagram.com', 'reddit.com',
}

def is_trusted_domain(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    return any(
        hostname == td or hostname.endswith(f".{td}")
        for td in TRUSTED_DOMAINS
    )

def subdomain_count(url):
    hostname = urlparse(url).hostname or ''
    return hostname.count('.') - 1
def has_suspicious_words(url):
    keywords = ['login', 'secure', 'account', 'update', 'free', 'verify', 'password', 'ebayisapi', 'banking', 'signin']
    return int(any(word in url.lower() for word in keywords))
def url_path_length(url):
    return len(urlparse(url).path)

def extract_features(url):
    return {
        'url_len': len(url),
        'abnormal_url': abnormal_url(url),
        'count_dot_hostname': dot_count_hostname(url),
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
        'trusted_domain': is_trusted_domain(url),
        'subdomain_count': subdomain_count(url),
        'suspicious_words': has_suspicious_words(url),
        'url_entropy': calculate_entropy(url),
        'url_path_length': url_path_length(url),
    }

# ====== Prediction Function ======
def predict_url(url):
    features_dict = extract_features(url)
    features_df = pd.DataFrame([features_dict])
    features_df = features_df[features]  # Align with training order

    # XGBoost prediction
    xgb_pred = xgb_model.predict(features_df)[0]
    xgb_label = 'Phishing' if xgb_pred == 1 else 'Legitimate'
    xgb_confidence = max(xgb_model.predict_proba(features_df)[0])

    # Fusion prediction
    fusion_pred = fusion_model.predict(features_df)[0]
    fusion_label = 'Phishing' if fusion_pred == 1 else 'Legitimate'
    fusion_confidence = max(fusion_model.predict_proba(features_df)[0])

    return {
        "xgb": {"label": xgb_label, "confidence": xgb_confidence},
        "fusion": {"label": fusion_label, "confidence": fusion_confidence}
    }

# ====== Final Logic ======
if __name__ == "__main__":
    url = input("Enter the URL to check: ").strip()
    if not url:
        print("❗ No URL entered.")
    else:
        result = predict_url(url)
        is_flagged_by_google = check_google_safe_browsing(url)

        # ✅ Show Google's judgment
        print(f"🔐 Google Safe Browsing Flagged: {'Yes' if is_flagged_by_google else 'No'}")

        if is_flagged_by_google:
            final_label = "Phishing (⚠️ Flagged by Google Safe Browsing)"
        else:
            final_label = result['fusion']['label']

        print(f"\n🔎 Final Result: {final_label}")
        print(f"  - XGBoost: {result['xgb']['label']} ({result['xgb']['confidence']:.2f})")
        print(f"  - Fusion : {result['fusion']['label']} ({result['fusion']['confidence']:.2f})")
