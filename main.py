'''
project/
│
├── .env
├── .vscode/
│   └── launch.json
│
├── main.py
├── config.py
├── requirements.txt
│
├── data_sources/
│   ├── __init__.py
│   ├── kaggle_source.py
│   └── gdrive_source.py
|   └── url_source.py
│
└── utils/
    ├── __init__.py
    |── apriori_analyzer.py
    |── cache_manager.py
    ├── debug_utils.py
    └── menu_utils.py
    └── file_checker.py
    └── file_manager.py
    └── kmeans_analyzer.py3
    |── linier_regresion_analyzer.py
    └── file_handler.py'''

from utils.menu_utils import display_menu
from utils.debug_utils import logger
from utils.file_manager import list_local_datasets, download_and_extract_zip, get_local_dataset_path, delete_local_dataset
from data_sources.kaggle_source import (
    setup_kaggle_api,
    download_kaggle_dataset,
    search_kaggle_dataset,
    preview_kaggle_dataset
)
from data_sources.url_source import download_from_url, preview_from_url
from data_sources.gdrive_source import download_from_gdrive, preview_from_gdrive
from utils.kmeans_analyzer import analyze_kmeans 
import config

from utils.apriori_analyzer import analyze_apriori
from utils.ensemble_analyzer import analyze_ensemble

def main():
    if not setup_kaggle_api(config.KAGGLE_USERNAME, config.KAGGLE_KEY):
        print("❌ Gagal inisialisasi Kaggle API. Cek konfigurasi di file .env atau config.py.")
        return

    while True: 
        choice = display_menu()
 
        if choice == "1":
            url = input("Masukkan URL dataset (ZIP/CSV/XLSX/JSON): ").strip()
            preview_from_url(url or "https://drive.google.com/file/d/1j5yCAUG5ooaOMFmrUjW0jNq90qQwrEjB/view?usp=sharing")
        elif choice == "2":
            query = input("Masukan direct link Gdrive (contoh: https://drive.google.com/uc?id=1j5yCAUG5ooaOMFmrUjW0jNq90qQwrEjB) atau tekan enter untuk dataset default: ").strip()
            preview_from_gdrive(query or "1j5yCAUG5ooaOMFmrUjW0jNq90qQwrEjB")
        elif choice == "3":
            query = input("Masukkan kata kunci pencarian dataset Kaggle: ").strip()
            limit_input = input("Masukkan limit pencarian (default 5) ").strip()
            if query:
                try:
                    limit = int(limit_input) if limit_input else 50
                except ValueError:
                    print("⚠️ Limit harus berupa angka. Proses menggunakan default (5).")
                    limit = 50
            search_kaggle_dataset(query,limit=int(limit))

        elif choice == "4":
            dataset = input("Masukkan nama dataset Kaggle (contoh: zynicide/wine-reviews): ").strip()
            preview_kaggle_dataset(dataset or "muhammadrezkyananda/data-rumah-makan-prima-kendari")

        #elif choice == "5":
        #    dataset = input("Masukkan ID dataset Kaggle untuk diunduh dan diekstrak(contoh: zynicide/wine-reviews): ").strip()
        #    if dataset:
        #        folder_path = download_and_extract_zip(dataset or "muhammadrezkyananda/data-rumah-makan-prima-kendari")
        #        if folder_path:
        #            print(f"📂 Dataset berhasil disimpan di: {folder_path}")

        elif choice == "5":
            list_local_datasets(show_files=True)

        elif choice == "6":
            datasets = list_local_datasets(show_files=False)
            if not datasets:
                print("(Belum ada dataset lokal untuk dianalisis.)")
                continue

            try:
                idx = int(input("Pilih dataset untuk analisis [1-n]: "))
                if idx < 1 or idx > len(datasets):
                    print("Nomor tidak valid.")
                    continue

                dataset_path = datasets[idx - 1].lstrip("/")
                print(f"Memilih dataset: {dataset_path}")
                if not dataset_path:
                    print("Dataset tidak ditemukan secara lokal.")
                    continue

                analyze_kmeans(dataset_path)

            except ValueError:
                print("Input tidak valid.")
                continue

        elif choice == "7":
            #file_path = input("Masukkan path file CSV/XLSX untuk analisis Apriori: ").strip()
            #path = input("Masukkan path dataset lokal (mis. data/local_datasets/.../file.csv): ")
            #filename = path.split("/")[-1]
            #analyze_with_apriori(path, filename)
            datasets = list_local_datasets(show_files=False)
            if not datasets:
                print("(Belum ada dataset lokal untuk dianalisis.)")
                continue

            try:
                idx = int(input("Pilih dataset untuk analisis [1-n]: "))
                if idx < 1 or idx > len(datasets):
                    print("Nomor tidak valid.")
                    continue

                dataset_path = datasets[idx - 1].lstrip("/")
                print(f"Memilih dataset: {dataset_path}")
                if not dataset_path:
                    print("Dataset tidak ditemukan secara lokal.")
                    continue

                analyze_apriori(dataset_path)

            except ValueError:
                print("Input tidak valid.")
                continue

        elif choice == "8":
            datasets = list_local_datasets(show_files=False)
            if not datasets:
                print("(Belum ada dataset lokal untuk dianalisis.)")
                continue

            try:
                idx = int(input("Pilih dataset untuk analisis [1-n]: "))
                if idx < 1 or idx > len(datasets):
                    print("Nomor tidak valid.")
                    continue

                dataset_path = datasets[idx - 1].lstrip("/")
                print(f"Memilih dataset: {dataset_path}")
                if not dataset_path:
                    print("Dataset tidak ditemukan secara lokal.")
                    continue

                analyze_ensemble(dataset_path)

            except ValueError:
                print("Input tidak valid.")
                continue
            
        elif choice == "9":
            delete_local_dataset()
        
        elif choice == "q":
            logger.info("Program dihentikan oleh pengguna.")
            print("👋 Keluar dari aplikasi...")
            break

        else:
            print("❌ Pilihan tidak valid. Coba lagi.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Dibatalkan oleh pengguna.")

