# utils/menu_utils.py

def display_menu():
    print("\n=== ANALISA DATASET ===")
    print("1. Dataset dari URL")
    print("2. Dataset dari Gdrive")
    print("3. Cari dataset di Kaggle")
    print("4. Download & Extract dataset ZIP dari Kaggle")
    print("5. Daftar dataset lokal yang telah diunduh")
    print("6. Analisis dataset dengan K-Means Clustering")
    print("7. Analisis dataset dengan Apriori (Association Rules)")
    print("8. Analisis dataset dengan Ensemble Methods")
    print("9. Hapus folder dataset lokal")
    print("Ketik 'q' untuk keluar.")
    return input("Pilih opsi [1-9]: ").strip()
