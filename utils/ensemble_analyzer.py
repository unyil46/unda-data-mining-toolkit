# utils/ensemble_analyzer.py
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier

from utils.debug_utils import logger
from utils.file_handler import detect_file_type
from utils.chart_utils import plot_distribution


def analyze_ensemble(dataset_path):

    selected_path = dataset_path
    local_path = selected_path.lstrip("/")

    print(f"\n🤖 Membaca dataset: {selected_path}")

    if not os.path.exists(local_path):
        logger.error("Dataset tidak ditemukan secara lokal.")
        print("❌ Dataset tidak ditemukan secara lokal.")
        return

    ext = detect_file_type(local_path)
    if not ext:
        logger.error(f"Format file {local_path} tidak dikenali.")
        print(f"❌ Format file {local_path} tidak dikenali.")
        return

    try:
        # ===============================
        # BACA DATASET
        # ===============================
        if ext == "csv":
            df = pd.read_csv(local_path)
        elif ext == "xlsx":
            df = pd.read_excel(local_path)
        elif ext == "json":
            df = pd.read_json(local_path)
        else:
            raise ValueError(f"Format {ext} belum didukung.")

        logger.info(f"Dataset dibaca: {df.shape[0]} baris, {df.shape[1]} kolom")
        print(f"Dataset berisi {df.shape[0]} baris dan {df.shape[1]} kolom")
        print(df.head())

        # ===============================
        # DETEKSI TARGET
        # ===============================
        candidate_targets = [ "target", "Target", "Outcome", "outcome","class", "Class","label", "Label","condition", "Condition"]
        target_col = next((c for c in candidate_targets if c in df.columns), None)
        if target_col is None:
            print("\nKolom tersedia:")
        for col in df.columns:
            print(f"- {col}")

        target_col = input("\nMasukkan nama kolom target: ").strip()

        if target_col not in df.columns:
            print("❌ Kolom target tidak valid.")
            return

        if target_col is None:
            print("❌ Kolom target tidak ditemukan.")
            return

        print(f"\n🎯 Kolom target terdeteksi: {target_col}")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ===============================
        # FITUR NUMERIK SAJA
        # ===============================
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[num_cols]

        if X.empty:
            print("❌ Tidak ada fitur numerik untuk ensemble.")
            return

        # ===============================
        # PREPROCESSING
        # ===============================
        preprocess = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ])

        # ===============================
        # SPLIT DATA
        # ===============================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ===============================
        # MODEL ENSEMBLE
        # ===============================
        rf = RandomForestClassifier(random_state=42)
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            random_state=42
        )
        gb = GradientBoostingClassifier(random_state=42)

        voting = VotingClassifier(
            estimators=[("rf", rf), ("ada", ada), ("gb", gb)],
            voting="soft"
        )

        model = Pipeline([
            ("preprocess", preprocess),
            ("ensemble", voting)
        ])

        # ===============================
        # TRAIN & EVALUATE
        # ===============================
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print("\n✅ Hasil Ensemble Methods")
        print("=========================")
        print(f"Akurasi : {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        logger.info(f"Akurasi Ensemble: {acc:.4f}")

    except Exception as e:
        logger.error(f"Gagal menjalankan Ensemble Methods: {e}")
        print(f"❌ Terjadi kesalahan: {e}")
