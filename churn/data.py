"""Load and clean the IBM Telco Customer Churn dataset (7,043 customers, public)."""
from pathlib import Path
import urllib.request

import pandas as pd

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
CACHE = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"


def download(refresh: bool = False) -> Path:
    if refresh or not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    return CACHE


def load() -> pd.DataFrame:
    """Return one row per customer with a binary `churn` target.

    TotalCharges is blank for 11 brand-new customers (tenure 0); they have been billed nothing yet,
    so the blank is set to 0 rather than dropped.
    """
    df = pd.read_csv(download())
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    df["churn"] = (df.pop("Churn") == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df.drop(columns="customerID")


NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NUMERIC + ["churn"]]
