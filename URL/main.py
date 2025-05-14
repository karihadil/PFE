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
import httpx
import asyncio

# === Load Model and Features ===
xgb_model = joblib.load("phishing_detector_xgb.pkl")
features = joblib.load("features.pkl")

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mail.google.com",
        "chrome-extension://ifbhjmmbmgoomddjcbimegfciahkldng"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    urls: List[str]

# === Trusted Domains and Keywords ===
TRUSTED_DOMAINS = {
    'google.com', 'github.com', 'apple.com', 'linkedin.com', 'paypal.com',
    'dropbox.com', 'youtube.com', 'openai.com', 'cloudflare.com', 'netflix.com',
    'microsoft.com', 'facebook.com', 'stackoverflow.com'
}
SUSPICIOUS_KEYWORDS = [
    'login', 'secure', 'account', 'update', 'free', 'verify', 'password',
    'ebayisapi', 'banking', 'signin'
]

GOOGLE_API_KEY = "AIzaSyBlIx-aWXKXyQ-tAGHsEOwtZexmb5AufTs"

async def check_google_safe_browsing(url):
    endpoint = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
    body = {
        "client": {"clientId": "phishing-detector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    params = {"key": GOOGLE_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.post(endpoint, params=params, json=body)
            result = response.json()
            return bool(result.get("matches"))
    except Exception as e:
        print(f"[ERROR] GSB check failed: {e}")
        return False

def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path
    ext = tldextract.extract(url)
    return {
        'url_len': len(url),
        'abnormal_url': 0 if ext.domain in url else 1,
        'count_dot_hostname': hostname.count('.'),
        'count@': url.count('@'),
        'special_chars_count': sum(url.count(c) for c in "@-_%=&?"),
        'https': int('https' in url.lower()),
        'domain_len': len(ext.domain),
        'count_dir': path.count('/'),
        'short_url': int(bool(re.search(r'bit\\.ly|goo\\.gl|tinyurl|ow\\.ly|t\\.co', hostname))),
        'count-https': url.count('https'),
        'count-http': url.count('http'),
        'count%': url.count('%'),
        'count-': url.count('-'),
        'count=': url.count('='),
        'hostname_len': len(hostname),
        'fd_length': len(path.split('/')[1]) if len(path.split('/')) > 1 else 0,
        'tld_len': len(ext.suffix),
        'count-digits': sum(c.isdigit() for c in url),
        'count-letters': sum(c.isalpha() for c in url),
        'trusted_domain': int(any(hostname.endswith(td) for td in TRUSTED_DOMAINS)),
        'subdomain_count': hostname.count('.') - 1,
        'suspicious_words': int(any(k in url.lower() for k in SUSPICIOUS_KEYWORDS)),
        'url_path_length': len(path),
        'url_entropy': -sum((c / len(url)) * math.log2(c / len(url)) for c in Counter(url).values() if c > 0)
    }

@app.post("/predict")
async def predict(request: URLRequest):
    urls = request.urls
    results = []

    async def analyze_url(url):
        feat = extract_features(url)
        df = pd.DataFrame([feat])[features]

        xgb_proba = xgb_model.predict_proba(df)[0]
        phishing_conf = xgb_proba[1]
        xgb_pred = 1 if phishing_conf >= 0.5 else 0

        google_flagged = await check_google_safe_browsing(url)

        if google_flagged:
            final_label = "Phishing (⚠️ Flagged by Google Safe Browsing)"
        elif xgb_pred == 1:
            if phishing_conf >= 0.65:
                final_label = "Phishing (High Confidence)"
            else:
                final_label = "Suspicious (Low Confidence)"
        else:
            final_label = "Legitimate"

        return {
            "url": url,
            "xgb": {
                "label": "Phishing" if xgb_pred else "Legitimate",
                "confidence": float(round(phishing_conf, 2))
            },
            "google": google_flagged,
            "final_label": final_label
        }

    results = await asyncio.gather(*(analyze_url(url) for url in urls))
    return {"results": results}
