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

# Load the saved model and features
xgb_model = joblib.load('phishing_detector_xgb.pkl')
features = joblib.load('features.pkl')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://ifbhjmmbmgoomddjcbimegfciahkldng",
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

# Feature extraction functions
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

@app.post("/predict")
async def predict_phishing(request: URLRequest):
    results = []
    for url in request.urls:
        features_dict = extract_features(url)
        features_df = pd.DataFrame([features_dict])
        prediction = xgb_model.predict(features_df)[0]
        prediction_label = 'Phishing' if prediction == 1 else 'Legitimate'
        results.append(URLResult(url=url, prediction=prediction_label))
    return {"results": results}