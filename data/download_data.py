"""
Downloads the Telco Customer Churn dataset from a public source.
Run once: python data/download_data.py
"""
import urllib.request
import os

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = "data/churn.csv"

os.makedirs("data", exist_ok=True)
print(f"Downloading dataset...")
urllib.request.urlretrieve(URL, OUT)
print(f"Saved to {OUT}")