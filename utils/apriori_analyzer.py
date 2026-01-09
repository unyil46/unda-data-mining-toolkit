# utils/apriori_analyzer.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from utils.file_handler import read_full_file
from utils.debug_utils import logger


def analyze_apriori(source, filename=None, min_support=0.05, metric="lift", min_threshold=1.0):
    """
    Melakukan analisis asosiasi menggunakan algoritma Apriori.
    Parameter:
      - source: path lokal atau URL dataset
      - filename: nama file untuk deteksi ekstensi
      - min_support: ambang minimum support
      - metric: metrik untuk aturan asosiasi ('lift', 'confidence', dst)
      - min_threshold: nilai minimum metrik
    """
    try:
        logger.info(f"Membaca dataset untuk analisis Apriori: {filename or source}")
        df = read_full_file(source, filename)

        # --- Preprocessing ---
        # Coba deteksi apakah dataset sudah berbentuk 0/1 (one-hot)
        if df.dtypes.isin(['object', 'bool']).any():
            logger.info("Mendeteksi kolom non-numerik, akan dilakukan one-hot encoding otomatis.")
            df = pd.get_dummies(df.astype(str))

        if not ((df == 0) | (df == 1)).all().all():
            logger.warning("Dataset tidak biner sepenuhnya. Nilai akan dibinarisasi (>0 → 1, sisanya 0).")
            df = (df > 0).astype(int)

        logger.info("Menjalankan algoritma Apriori...")
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            print("⚠️ Tidak ditemukan itemset yang memenuhi ambang support.")
            return None

        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        rules = rules.sort_values(by="lift", ascending=False)

        print("\n=== HASIL ANALISIS APRIORI ===")
        print(f"Jumlah aturan yang ditemukan: {len(rules)}")
        print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).to_string(index=False))

        logger.info(f"Analisis Apriori selesai: {len(rules)} aturan ditemukan.")
        return rules

    except Exception as e:
        logger.error(f"Gagal melakukan analisis Apriori: {e}")
        print(f"(Gagal melakukan analisis Apriori: {e})")
        return None
