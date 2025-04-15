import numpy as np
import pandas as pd
import math
from collections import Counter
import re
from urllib.parse import urlparse
import tldextract
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ipaddress import ip_address
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('C:\\Users\\DELL\\OneDrive\\Bureau\\PFE\\backend\\app\\models\\dataset_phishing.csv')
print(df.head())
print(df.info())
print(df.describe())

df1 = [
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_slash",
    "nb_qm",
    "nb_percent",
    "nb_www",
    "nb_star",
    "nb_subdomains",
    "longest_word_path",
    "ratio_digits_url",
    "avg_word_path",
    "phish_hints",
    "prefix_suffix",
    "abnormal_subdomain",
    "tld_in_path",
    "tld_in_subdomain",
    "nb_space",
    "nb_underscore",
    "random_domain"
]

filtered_df  = df[df1 + ["status"]].dropna().drop_duplicates().reset_index(drop=True)
print(filtered_df.info())
correlation_matrix = filtered_df[df1].corr()
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of URL-based Features")
plt.tight_layout()
plt.show()

x=filtered_df.drop(columns=["status"])
print(x.head())
encoder = LabelEncoder()
filtered_df["status"] = encoder.fit_transform(filtered_df["status"])
y = filtered_df["status"]
print(y.value_counts())
plt.figure()
plt.bar(y.value_counts().index, y.value_counts(), color=["green", "red"])
plt.xlabel('Status')
plt.ylabel('Count')
plt.title('Distribution of URL Status')
plt.tight_layout()
plt.show()
categorical_cols = x.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in categorical_cols:
    x[col] = encoder.fit_transform(x[col])
print(x.dtypes)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train,y_train)
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": x.columns, "Importance": importances})
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance_df["Importance"], y=feature_importance_df["Feature"])
plt.title("Top 10 Feature Importances in Detecting Phishing URLs")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
print("rfffffffffffffff")
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("xgbbbbbbbbbbbbbbbbbbbb")
import optuna
xgb=XGBClassifier(n_estimators=100, random_state=42, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)
xgb.fit(x_train,y_train)
y_pred=xgb.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


rs = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    {
    'n_estimators':     [100,200,300,400,500,600],
    'max_depth':        [None,5,10,20,30,40],
    'min_samples_split':[2,5,10,15],
    'min_samples_leaf': [1,2,4,6],
    'max_features':     [None, 'sqrt', 'log2']      # ← replaced 'auto' with None
    },
    n_iter=50,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

rs.fit(x_train, y_train)
best_rf = rs.best_estimator_
if isinstance(best_rf, RandomForestClassifier):
    y_pred = best_rf.predict(x_test)
else:
    raise TypeError("The best estimator is not a RandomForestClassifier and cannot be used for prediction.")
print("Best Random Forest Parameters:", rs.best_params_)
print("Best Random Forest Score:", rs.best_score_)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
from sklearn.model_selection import learning_curve
# Reduce train_sizes granularity
train_sizes=np.linspace(0.1, 0.8, 5)

def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, scoring="accuracy", n_jobs=-1):
    train_sizes, train_scores, val_scores, fit_times, score_times = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color='blue')
    
    plt.plot(train_sizes, val_scores_mean, 'o-', color='green', label='Cross-validation score')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std, alpha=0.2, color='green')

    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel(scoring.capitalize())
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Plot learning curve for your best Random Forest model
# Instead of n_jobs=-1 (use all CPUs), try using fewer workers
plot_learning_curve(best_rf, x_train, y_train, title="Learning Curve - Random Forest", scoring="f1", n_jobs=2)
