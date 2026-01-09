# utils/kmeans_analyzer.py
import os
import pandas as pd
from sklearn.cluster import KMeans
from utils.debug_utils import logger
from utils.file_handler import detect_file_type
from utils.file_handler import read_file_preview  # gunakan fungsi pembaca umum
from utils.file_manager import list_local_datasets
from utils.chart_utils import plot_kmeans_clusters,plot_linear_regression,plot_apriori_support,plot_distribution
def analyze_kmeans(dataset_path):

    selected_path = dataset_path
    local_path = selected_path.lstrip("/")  # ubah ke9 path relatif sistem
    print(f"\n Membaca dataset: {selected_path}")

    if not os.path.exists(local_path):
        logger.error("Dataset tidak ditemukan secara lokal.")
        print("Dataset tidak ditemukan secara lokal.")
        return

    # Deteksi format file
    ext = detect_file_type(local_path)
    if not ext:
        logger.error(f"Format file {local_path} tidak dikenali.")
        print(f" Format file {local_path} tidak dikenali.")
        return

    try:
        # Gunakan pembacaan penuh berdasarkan ekstensi
        if ext == "csv":
            df = pd.read_csv(local_path)
        elif ext == "xlsx":
            df = pd.read_excel(local_path)
        elif ext == "json":
            df = pd.read_json(local_path)
        else:
            raise ValueError(f"Format file {ext} belum didukung untuk pembacaan penuh.")

        logger.info(f"Dataset {local_path} berhasil dibaca ({df.shape[0]} baris, {df.shape[1]} kolom).")

        print(f"\nDataset berisi {df.shape[0]} baris dan {df.shape[1]} kolom.")
        print("Menampilkan 5 baris pertama:")
        print(df.head())

        # Pilih kolom numerik untuk clustering
        numeric_df = df.select_dtypes(include=['number'])
        # Tangani missing value
        if numeric_df.isnull().any().any():
            logger.warning("Dataset mengandung nilai kosong. Mengisi dengan median.")
            numeric_df = numeric_df.fillna(numeric_df.median())

        if numeric_df.empty:
            print(" Tidak ada kolom numerik untuk analisis K-Means.")
            return

        print("\nKolom numerik yang digunakan:")
        print(", ".join(numeric_df.columns))

        # Input jumlah cluster
        try:
            n_clusters = int(input("\nMasukkan jumlah cluster (default=3): ") or 3)
        except ValueError:
            n_clusters = 3

        # Jalankan K-Means
        model = KMeans(n_clusters=n_clusters, random_state=42)
        df["Cluster"] = model.fit_predict(numeric_df)
        plot_kmeans_clusters(df, x_col=numeric_df.columns[0], y_col=numeric_df.columns[1], title="Hasil K-Means Clustering")

        print(f"\nAnalisis K-Means selesai. Total cluster: {n_clusters}")
        print(df[["Cluster"] + list(numeric_df.columns)].head())

        output_path = local_path.replace(".csv", "_clustered.csv")
        df.to_csv(output_path, index=False)
        print(f"\n Hasil disimpan ke: {output_path}")
        logger.info(f"Hasil K-Means disimpan ke {output_path}")

    except Exception as e:
        logger.error(f"Gagal melakukan analisis K-Means: {e}")
        print(f" Terjadi kesalahan saat analisis: {e}")

