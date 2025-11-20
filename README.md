# unda_ml

Aplikasi analisis data dan machine learning berbasis Python CLI. Mendukung pengambilan data dari Kaggle, Google Drive, atau URL serta metode analisis K-Means dan Apriori.

## Fitur Utama

- Import data dari Kaggle, Google Drive, dan URL
- Analisis K-Means Clustering
- Analisis Apriori Association
- Otomatisasi pengelolaan file data

## Cara Menggunakan di Google Colab

### Step 1: Setup Environment

Copy-paste code ini di cell pertama:

```python
# Clone repository dan install dependencies
!git clone https://github.com/unyil46/unda_ml.git
%cd unda_ml
!pip install -q -r requirements.txt

# Setup Kaggle API (gunakan placeholder untuk dataset publik)
import os
os.environ['KAGGLE_USERNAME'] = 'your_username_here'
os.environ['KAGGLE_KEY'] = 'your_api_key_here'
```

### Step 2: Jalankan Aplikasi

Di cell berikutnya, jalankan:

```python
!python main.py
```

> ℹ️ Aplikasi bisa digunakan langsung tanpa mengganti API key untuk mengakses dataset Kaggle publik. Untuk fitur lengkap, ganti dengan [API key Kaggle Anda](https://www.kaggle.com/settings).

## Menu Aplikasi

Setelah menjalankan aplikasi, Anda akan melihat menu berikut:
```
=== ANALISA DATASET ===
1. Dataset dari URL
2. Dataset dari Gdrive
3. Cari dataset di Kaggle
4. Download & Extract dataset ZIP dari Kaggle
5. Daftar dataset lokal yang telah diunduh
6. Analisis dataset dengan K-Means Clustering
7. Analisis dataset dengan Apriori (Association Rules)
8. Hapus folder dataset lokal
Ketik 'q' untuk keluar.
Pilih opsi [1-8]:
```

## Contoh Import Data

### Google Drive

Cara mendapatkan File ID:
- Copy link dari Google Drive
- Ambil bagian `FILE_ID` pada format `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`

Contoh:
```
File ID: 1j5yCAUG5ooaOMFmrUjW0jNq90qQwrEjB
```

### URL Langsung

Format yang didukung:
- CSV: https://example.com/data.csv
- Excel: https://example.com/data.xlsx
- JSON: https://example.com/data.json
- ZIP: https://example.com/dataset.zip

Contoh:
```
URL: https://raw.githubusercontent.com/username/repo/main/data.csv
```

## Fitur Dataset

Bisa digunakan tanpa API key:
- Cari & download dataset Kaggle publik
- Import dari Google Drive
- Import dari URL langsung

Butuh API key asli:
- Private datasets & competitions

## Struktur Folder

```
unda_ml/
├── main.py
├── config.py
├── requirements.txt
├── data_sources/
├── utils/
└── data/
```

## Kontribusi

Fork repo, buat branch dan PR untuk penambahan fitur atau perbaikan.

## Lisensi

MIT
