# utils/linier_regresion_analyzer.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.debug_utils import logger
from utils.file_handler import detect_file_type
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def analyze_linear_regression(dataset_path):
    """
    Melakukan analisis regresi linier sederhana/multivariat.
    User memilih target kolom Y dan fitur X dari dataset numerik.
    """
    local_path = dataset_path.lstrip("/")
    print(f"\n Membaca dataset: {dataset_path}")

    if not os.path.exists(local_path):
        print(" Dataset tidak ditemukan secara lokal.")
        logger.error("Dataset tidak ditemukan.")
        return

    try:
        ext = detect_file_type(local_path)
        if ext == "csv":
            df = pd.read_csv(local_path)
        elif ext == "xlsx":
            df = pd.read_excel(local_path)
        elif ext == "json":
            df = pd.read_json(local_path)
        else:
            print(f" Format file {ext} belum didukung.")
            return

        numeric_df = df.select_dtypes(include=["number"]).dropna()
        if numeric_df.empty:
            print(" Tidak ada kolom numerik yang bisa dianalisis.")
            return

        print("\nKolom numerik yang tersedia:")
        for i, col in enumerate(numeric_df.columns, 1):
            print(f"{i}. {col}")

        target_col = input("\nMasukkan nama kolom target (Y): ").strip()
        if target_col not in numeric_df.columns:
            print(" Kolom target tidak valid.")
            return

        feature_cols = [c for c in numeric_df.columns if c != target_col]
        print(f" Kolom fitur yang digunakan: {', '.join(feature_cols)}")

        X = numeric_df[feature_cols]
        y = numeric_df[target_col]

        # Split data train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model regresi linier
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n Hasil Analisis Regresi Linier:")
        print(f"Koefisien: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
        print(f"MSE (Mean Squared Error): {mse:.4f}")
        print(f"R² Score: {r2:.4f}")

        output_path = local_path.replace(".csv", "_regression_result.csv")
        pd.DataFrame({"y_actual": y_test, "y_predicted": y_pred}).to_csv(output_path, index=False)
        print(f"\n Hasil prediksi disimpan ke: {output_path}")

        logger.info(f"Hasil regresi linier disimpan ke {output_path}")

    except Exception as e:
        logger.error(f"Gagal melakukan analisis regresi linier: {e}")
        print(f" Terjadi kesalahan: {e}")
