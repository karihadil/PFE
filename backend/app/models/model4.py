from cProfile import label
import pandas as pd
import re
from urllib.parse import urlparse
import tldextract
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tld import get_tld
import os.path
df = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Bureau\\PFE\\backend\\app\\models\\filtered_dataset.csv')
print(df.head()) 
print(df.isna().sum())
df['url_len'] = [len(url) for url in df.url]
print(df['url_len'].head())
#the distribution of url length depending on the type of url (legit or phishing)
sns.displot(df, x='url_len', hue='status', kind='kde', fill=True)
plt.xlabel('URL Length')
plt.ylabel('Density')
plt.title('Distribution of URL Length by Type')
plt.show()
encoder=LabelEncoder()
df["status_code"]=encoder.fit_transform(df["status"]) #0 legit 1 phishing
print(df.head())
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

