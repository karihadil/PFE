from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import math
from urllib.parse import urlparse
import tldextract
import re
from collections import Counter

# ====== Load Updated Model and Features ======
xgb_model = joblib.load('phishingxgb.pkl')
features = joblib.load('features_list.pkl')

# ====== FastAPI Setup ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://bfcjiigpkmcpimmhegkaaiidneieiklp",
        "https://mail.google.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    urls: List[str]

class URLResult(BaseModel):
    url: str
    prediction: str  # 'Phishing' or 'Legitimate'

# ====== Feature Extraction ======
TRUSTED_DOMAINS = {
}

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
    match = re.search(r'bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co', domain)
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

def url_path_length(url):
    return len(urlparse(url).path)

def subdomain_count(url):
    hostname = urlparse(url).hostname or ''
    return hostname.count('.') - 1

def has_suspicious_words(url):
    keywords = ['login', 'secure', 'account', 'update', 'free', 'verify', 'password', 'ebayisapi', 'banking', 'signin']
    return int(any(word in url.lower() for word in keywords))
def is_trusted_domain(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    return any(
        hostname == td or hostname.endswith(f".{td}")
        for td in TRUSTED_DOMAINS
    )

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
        'url_path_length': url_path_length(url),
        'url_entropy': calculate_entropy(url)
    }

# ====== Prediction Endpoint ======
@app.post("/predict")
async def predict_phishing(request: URLRequest):
    results = []
    for url in request.urls:
        features_dict = extract_features(url)
        features_df = pd.DataFrame([features_dict])[features]
        prediction = xgb_model.predict(features_df)[0]
        label = "Phishing" if prediction == 1 else "Legitimate"
        results.append(URLResult(url=url, prediction=label))
    return {"results": results}
