import os
import re
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect
from typing import List, Dict, Tuple
from collections import Counter

class DataPreprocessor:
    def __init__(self):
        self.stopwords = set(["the", "is", "and", "of", "to", "a"])  # extend later

    # ------------------- Step 1: Load -------------------
    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
        return df

    def load_pdf(self, path: str) -> List[str]:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        texts = [page.extract_text() for page in reader.pages if page.extract_text()]
        print(f"Loaded {len(texts)} pages from {path}")
        return texts

    # ------------------- Step 2: Clean Text -------------------
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)  # normalize whitespace
        text = re.sub(r"http\S+", "", text)  # remove URLs
        text = re.sub(r"[^A-Za-z0-9\s.,?!%₹-]", "", text)  # keep only useful chars
        return text

    # ------------------- Step 3: Deduplicate -------------------
    def hash_text(self, text: str) -> str:
        return hashlib.md5(text.lower().encode()).hexdigest()

    def deduplicate(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        seen = set()
        rows = []
        for _, row in df.iterrows():
            combined = " ".join([str(row[c]) for c in text_cols])
            h = self.hash_text(combined)
            if h not in seen:
                seen.add(h)
                rows.append(row)
        print(f"Deduplicated from {len(df)} → {len(rows)}")
        return pd.DataFrame(rows)

    # ------------------- Step 4: Language Detection -------------------
    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            if lang.startswith("ml"):
                return "malayalam"
            return "english"
        except Exception:
            return "unknown"

    def add_language_column(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        df["language"] = df[text_col].apply(self.detect_language)
        return df

    # ------------------- Step 5: EDA -------------------
    def eda_report(self, df: pd.DataFrame, question_col: str, answer_col: str, save_dir="eda_report"):
        os.makedirs(save_dir, exist_ok=True)

        print("\n📊 Basic Info:")
        print(df.info())
        print("\n🔍 Missing Values:")
        print(df.isna().sum())
        print("\n🔄 Duplicate rows:", df.duplicated().sum())
        print("\n🌍 Language distribution:")
        print(df["language"].value_counts())

        # Plot language distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x="language", data=df)
        plt.title("Language Distribution")
        plt.savefig(os.path.join(save_dir, "language_distribution.png"))
        plt.close()

        # Length distribution
        df["q_len"] = df[question_col].apply(lambda x: len(str(x).split()))
        df["a_len"] = df[answer_col].apply(lambda x: len(str(x).split()))

        plt.figure(figsize=(8, 5))
        sns.histplot(df["q_len"], bins=30, color="blue", alpha=0.7, label="Question")
        sns.histplot(df["a_len"], bins=30, color="green", alpha=0.5, label="Answer")
        plt.legend()
        plt.title("Text Length Distribution")
        plt.savefig(os.path.join(save_dir, "text_length_distribution.png"))
        plt.close()

        print(f"📂 EDA report saved to {save_dir}/")

    # ------------------- Step 6: Normalize Dataset -------------------
    def prepare_dataset(self, df: pd.DataFrame, q_col="question", a_col="answer", source="faq") -> pd.DataFrame:
        df = df[[q_col, a_col]].copy()
        df[q_col] = df[q_col].apply(self.clean_text)
        df[a_col] = df[a_col].apply(self.clean_text)

        df = self.deduplicate(df, [q_col, a_col])
        df = self.add_language_column(df, q_col)
        df["source"] = source
        return df

    # ------------------- Step 7: Save -------------------
    def save_cleaned(self, df: pd.DataFrame, path="cleaned_dataset.csv"):
        df.to_csv(path, index=False)
        print(f"✅ Clean dataset saved to {path}")
