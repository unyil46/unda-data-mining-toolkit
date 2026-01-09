# utils/file_manager.py
import os
import zipfile
import tempfile
import shutil
import subprocess
import requests
import logging
import zipfile
import sys
import os
import shutil
import stat
from utils.debug_utils import logger
from utils.file_handler import read_file_preview, detect_file_type
from requests.utils import urlparse

DATA_DIR = os.path.join("data", "local_datasets")
SUPPORTED_EXT = ["csv", "json", "xlsx"]


CACHE_DIR = os.path.join(".cache", "preview")
os.makedirs(CACHE_DIR, exist_ok=True)


logger = logging.getLogger(__name__)


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR

def _sanitize_name(name: str) -> str:
    """Ubah nama dataset menjadi aman untuk nama folder."""
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")

def download_and_extract_zip(url_or_dataset: str, output_dir: str = None):
    """
    Unduh dataset dari URL atau Kaggle, lalu ekstrak ke folder:
      data/local_datasets/<nama_dataset>/

    Jika file bukan ZIP, tetap disimpan di folder tersebut.
    Return:
        tuple: (path_folder_dataset, daftar_file)
    """
    ensure_data_dir()

    # Tentukan nama folder dataset
    if url_or_dataset.startswith("http"):
        dataset_name = _sanitize_name(os.path.splitext(os.path.basename(urlparse(url_or_dataset).path))[0])
    else:
        dataset_name = _sanitize_name(url_or_dataset.split("/")[-1])

    dataset_folder = os.path.join(DATA_DIR, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)

    logger.info(f"Mengunduh dataset: {url_or_dataset}")
    logger.info(f"📂 Folder tujuan: {dataset_folder}")

    # Unduh dari URL
    if url_or_dataset.startswith("http"):
        file_name = os.path.basename(urlparse(url_or_dataset).path)
        file_path = os.path.join(dataset_folder, file_name)
        with requests.get(url_or_dataset, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Jika file adalah ZIP → ekstrak
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
            os.remove(file_path)
            logger.info(f"ZIP diekstrak ke: {dataset_folder}")
        else:
            logger.info(f"File bukan ZIP, disimpan langsung di: {dataset_folder}")

    # Unduh dari Kaggle
    else:
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", url_or_dataset, "-p", dataset_folder, "--unzip"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Gagal mengunduh dataset dari Kaggle: {e}")
            return None, []

    # Daftar semua file hasil unduhan
    files = [
        f for f in os.listdir(dataset_folder)
        if os.path.isfile(os.path.join(dataset_folder, f))
    ]
    logger.info(f"Berhasil disimpan di: {dataset_folder}")
    logger.info(f"Berisi {len(files)} file: {', '.join(files) if files else '(kosong)'}")

    return dataset_folder, files

def download_and_preview_zip(source: str, metadata_text: str = None):
    """
    Unduh dataset ZIP dari URL atau Kaggle, ekstrak sementara, 
    tampilkan preview salah satu file, dan kembalikan hasil preview sebagai string.

    - Jika `source` diawali dengan 'http' → dianggap URL langsung ke ZIP.
    - Jika tidak → dianggap ID dataset Kaggle (misal: 'zynicide/wine-reviews').
    - File diekstrak ke folder sementara (tempfile.TemporaryDirectory).
    - Tidak menyimpan dataset ke /data/local_datasets.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "dataset.zip")

        # === Unduh file ===
        if source.startswith("http"):
            #logger.info(f"Mengunduh dataset dari URL: {source}")
            print(f"🌐 Mengunduh ZIP dari URL...\n{source}")
            with requests.get(source, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            #logger.info(f"ZIP URL disimpan sementara di: {zip_path}")

        else:
            #logger.info(f"Mengunduh dataset Kaggle: {source}")
            print(f"📦 Mengunduh dataset Kaggle...\n{source}")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", source, "-p", tmpdir],
                check=True
            )
            # Cari ZIP hasil unduhan Kaggle
            zips = [f for f in os.listdir(tmpdir) if f.endswith(".zip")]
            if not zips:
                print("❌ Tidak ada file ZIP ditemukan setelah unduh Kaggle.")
                return "(gagal unduh)"
            zip_path = os.path.join(tmpdir, zips[0])
            folder_name = os.path.splitext(os.path.basename(zip_path))[0]
            #logger.info(f"ZIP Kaggle disimpan sementara di: {zip_path}")

        # === Ekstrak ZIP ke folder sementara ===
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        #logger.info(f"ZIP diekstrak sementara di: {tmpdir}")

        # === Daftar file hasil ekstraksi ===
        #extracted_files = [
        #    f for f in os.listdir(tmpdir)
        #    if not os.path.isdir(os.path.join(tmpdir, f)) and not f.endswith(".zip")
        #]
        
        extracted_files = []
        for root, _, files in os.walk(tmpdir):
            for f in files:
                extracted_files.append(os.path.join(root, f))

        if not extracted_files:
            print("(ZIP kosong, tidak ada file data.)")
            return "(ZIP kosong)"

        print("\n=== File dalam ZIP ===")
        #for i, name in enumerate(extracted_files, 1):
        #    print(f"{i}. {name}")
        for i, path in enumerate(extracted_files, 1):
            rel_path = os.path.relpath(path, tmpdir)
            print(f"{i}. {rel_path}")
        
        # === Pilih file untuk preview ===
        try:
            choice = int(input("\nPilih file untuk preview [1-n]: "))
            if choice < 1 or choice > len(extracted_files):
                print("Nomor tidak valid.")
                return "(dibatalkan)"
        except ValueError:
            print("Input tidak valid.")
            return "(dibatalkan)"

        # === Preview file ===
        file_path = os.path.join(tmpdir, extracted_files[choice - 1])
        #logger.info(f"Preview file sementara: {file_path}")
        preview_text = read_file_preview(file_path, extracted_files[choice - 1]) 

        print("\n--- Cuplikan 5 baris pertama ---")
        print(preview_text)
        save_confirmation(file_path,folder_name)

        # === Simpan metadata sementara (jika ada) ===
        if metadata_text:
            meta_path = os.path.join(tmpdir, "metadata_preview.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(metadata_text + "\n\n--- Cuplikan Data ---\n")
                f.write(preview_text)
           # logger.info(f"Metadata sementara disimpan di: {meta_path}")

        # Folder tmp akan otomatis dihapus saat keluar blok `with`
        return preview_text

def _download_and_extract_zip(dataset: str):
    """
    Mengunduh dataset Kaggle (ZIP) dan mengekstraknya ke data/local_datasets/
    """
    ensure_data_dir()
    target_dir = os.path.join(DATA_DIR, dataset.replace("/", "_"))
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n📦 Mengunduh dan mengekstrak dataset '{dataset}' ...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", target_dir, "--unzip"], check=True)
        logger.info(f"Dataset {dataset} berhasil diunduh dan diekstrak ke {target_dir}")
        print(f"✅ Dataset berhasil diekstrak ke: {target_dir}")
        return target_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Gagal mengunduh dataset {dataset}: {e}")
        print(f"❌ Gagal mengunduh dataset {dataset}: {e}")
        return None
    
def ___download_and_preview_zip(dataset, metadata_text):
    """Unduh dataset ZIP, ekstrak, preview salah satu file, dan return hasil string."""
    import tempfile, zipfile, os, subprocess
    from utils.file_handler import read_file_preview
    from utils.debug_utils import logger

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", tmpdir, "--unzip"], check=True)

        extracted_files = [f for f in os.listdir(tmpdir) if not os.path.isdir(os.path.join(tmpdir, f))]
        if not extracted_files:
            print("(ZIP kosong, tidak ada file data.)")
            return "(ZIP kosong)"

        print("\nFile yang tersedia di ZIP:")
        for i, name in enumerate(extracted_files, 1):
            print(f"{i}. {name}")

        try:
            choice = int(input("\nPilih file untuk preview [1-n]: "))
            if choice < 1 or choice > len(extracted_files):
                print("Nomor tidak valid.")
                return "(dibatalkan)"
        except ValueError:
            print("Input tidak valid.")
            return "(dibatalkan)"

        file_path = os.path.join(tmpdir, extracted_files[choice - 1])
        logger.info(f"Preview file lokal: {file_path}")
        preview_text = read_file_preview(file_path, extracted_files[choice - 1])

        print("\n--- Cuplikan 5 baris pertama ---")
        print(preview_text)
        return preview_text

def __download_and_extract_zip(url_or_dataset: str, output_dir: str = None):
    """
    Unduh file ZIP dari Kaggle atau URL, lalu ekstrak ke direktori sementara atau `output_dir`.
    """
    tmpdir = tempfile.mkdtemp() if output_dir is None else output_dir

    if url_or_dataset.startswith("http"):
        import requests
        zip_path = os.path.join(tmpdir, "dataset.zip")
        logger.info(f"Mengunduh dataset dari URL: {url_or_dataset}")
        with requests.get(url_or_dataset, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        logger.info(f"Mengunduh dataset Kaggle: {url_or_dataset}")
        subprocess.run(["kaggle", "datasets", "download", "-d", url_or_dataset, "-p", tmpdir, "--unzip"], check=True)
        return tmpdir

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir)

    return tmpdir


def preview_extracted_file(directory: str):
    """Preview salah satu file CSV/JSON/Excel di folder hasil ekstraksi."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".csv", ".json", ".xlsx", ".xls")):
                file_path = os.path.join(root, file)
                return read_file_preview(file_path)
    return "Tidak ditemukan file yang dapat dipreview."

def list_local_datasets(show_files=True):
    """
    Menampilkan daftar file dataset lokal yang tersedia.
    - Mencari file dengan ekstensi didukung (csv, xlsx, json)
    - Menampilkan path relatif gaya aplikasi (/data/local_datasets/...)
    Parameter `show_files` disertakan untuk kompatibilitas lama (diabaikan).
    """
    base_dir = ensure_data_dir()
    datasets = []

    # Rekursif mencari semua file dataset yang valid
    for root, _, files in os.walk(base_dir):
        for f in files:
            ext = f.lower().split(".")[-1]
            if ext in SUPPORTED_EXT:
                rel_path = os.path.join(root, f).replace("\\", "/")
                datasets.append(f"/{rel_path}")

    if not datasets:
        print("📂 Belum ada file dataset lokal yang ditemukan.")
        logger.info("Tidak ada file dataset lokal ditemukan.")
        return []

    
    print(f"\n===  {len(datasets)} file tersedia di dataset lokal ===")
    for i, d in enumerate(datasets, 1):
        print(f"{i}. {d}")
   
    #logger.info(f"Menampilkan {len(datasets)} file dataset lokal.")
    return datasets

def get_local_dataset_path(dataset_name: str):
    """
    Mengembalikan path lengkap dari dataset lokal tertentu.
    """
    path = os.path.join(DATA_DIR, dataset_name)
    return path if os.path.exists(path) else None
def preview_file(file_path: str):
    """
    Menampilkan preview dataset dari file lokal (hasil unduhan dari GDrive, Kaggle, atau URL langsung).
    - Jika file ZIP → ekstrak ke tmp, tampilkan daftar file, dan preview file terpilih.
    - Jika file CSV/XLSX/JSON → langsung tampilkan preview.
    """
    logger.info(f"📦 Memproses file untuk preview: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ File tidak ditemukan: {file_path}")
        return None

    # Deteksi tipe file berdasarkan konten/file magic
    file_type = detect_file_type(file_path)
    logger.info(f"Tipe file terdeteksi: {file_type}")

    if file_type is None:
        print(f"❌ Format file tidak dikenali untuk preview otomatis.")
        return None

    # --- Jika ZIP: ekstrak lalu preview file di dalamnya
    if file_type == "zip":
        logger.info("ZIP file terdeteksi, mengekstrak isi...")
        extracted_dir, files = extract_zip_and_list(file_path)

        print("\n=== DAFTAR FILE DALAM ZIP ===")
        for i, f in enumerate(files, 1):
            print(f"[{i}] {f}")

        try:
            idx = int(input("\nPilih file untuk preview [1-n]: "))
            selected_file = files[idx - 1]
        except (ValueError, IndexError):
            print("❌ Pilihan tidak valid.")
            return None

        selected_path = os.path.join(extracted_dir, selected_file)
        logger.info(f"File dipilih untuk preview: {selected_path}")

        preview_text = read_file_preview(selected_path, selected_file)
        print("\n=== PREVIEW DATASET ===")
        print(f"File: {selected_file}")
        print("\n--- Cuplikan Data ---")
        print(preview_text)
        if sys.stdin.isatty():  # hanya kalau user bisa input
            save_confirmation(file_path)
        else:
            print("💡 Mode non-interaktif — dataset tidak disimpan otomatis.")
        return preview_text
        

    # --- Jika CSV, XLSX, JSON: langsung preview
    elif file_type in ["csv", "xlsx", "json"]:
        logger.info(f"Membaca dan menampilkan preview untuk: {file_path}")
        preview_text = read_file_preview(file_path, os.path.basename(file_path))
        print("\n=== PREVIEW DATASET ===")
        print(f"File: {file_path}")
        print("\n--- Cuplikan Data ---")
        print(preview_text)
        if sys.stdin.isatty():  # hanya kalau user bisa input
            save_confirmation(file_path)
        else:
            print("💡 Mode non-interaktif — dataset tidak disimpan otomatis.")
        return preview_text

    else:
        print(f"❌ Format file '{file_type}' tidak didukung untuk preview otomatis.")
        return None
def extract_zip_and_list(zip_path: str):
    """
    Mengekstrak file ZIP ke direktori sementara dan mengembalikan:
      (path_folder_ekstraksi, daftar_file_di_dalamnya)
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file tidak ditemukan: {zip_path}")

    tmp_dir = tempfile.mkdtemp(prefix="extract_")
    logger.info(f"📦 Mengekstrak ZIP ke folder sementara: {tmp_dir}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    # Ambil daftar file (non-folder)
    extracted_files = []
    for root, _, files in os.walk(tmp_dir):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), tmp_dir)
            extracted_files.append(rel_path)

    if not extracted_files:
        logger.warning("ZIP kosong — tidak ada file di dalamnya.")
    else:
        logger.info(f"🗂️ {len(extracted_files)} file ditemukan dalam ZIP.")

    return tmp_dir, extracted_files
def handle_zip_preview(local_path: str, metadata_text: str = None):
    """
    Menangani preview untuk file ZIP:
    - Mengekstrak file.
    - Menampilkan daftar isi.
    - Meminta user memilih salah satu file untuk preview.
    """
    logger.info("ZIP file terdeteksi, mengekstrak isi...")
    extracted_dir, files = extract_zip_and_list(local_path)

    if not files:
        print("❌ ZIP kosong atau tidak ada file yang bisa diproses.")
        return None

    print("\n=== DAFTAR FILE DALAM ZIP ===")
    for i, f in enumerate(files, 1):
        print(f"[{i}] {f}")

    # Memilih file untuk preview
    try:
        idx = int(input("\nPilih file untuk preview [1-n]: ").strip() or "1")
        selected_file = files[idx - 1]
    except (ValueError, IndexError):
        print("❌ Pilihan tidak valid.")
        return None

    selected_path = os.path.join(extracted_dir, selected_file)
    logger.info(f"File dipilih untuk preview: {selected_path}")

    preview_text = read_file_preview(selected_path, selected_file)
    print("\n=== PREVIEW DATASET ===")
    print(f"File: {selected_file}")
    print("\n--- Cuplikan Data ---")
    print(preview_text)

    save_confirmation(local_path)
    
    # === Simpan metadata sementara (jika ada) ===
    with tempfile.TemporaryDirectory() as tmpdir:
        if metadata_text:
            meta_path = os.path.join(tmpdir, "metadata_preview.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(metadata_text + "\n\n--- Cuplikan Data ---\n")
                f.write(preview_text)
            logger.info(f"Metadata sementara disimpan di: {meta_path}")
        return preview_text
def handle_direct_preview(local_path: str, file_type: str):
    """
    Menangani preview untuk file langsung (CSV, XLSX, JSON, dsb).
    """
    preview_text = read_file_preview(local_path, os.path.basename(local_path))
    print("\n=== PREVIEW DATASET ===")
    print(f"Tipe file: {file_type}")
    print("\n--- Cuplikan Data ---")
    print(preview_text)
    save_confirmation(local_path)
    return preview_text
def save_confirmation(file_path: str, folder_name: str = None):
    print("\n Preview dataset berhasil ditampilkan.")
    
    #logger.info(f"Menampilkan konfirmasi penyimpanan untuk file: {file_path}")

    # Konfirmasi dari user
    confirm = input("Apakah Anda ingin menyimpan dataset ini ke folder lokal (data/local_datasets)? (y/n): ").lower().strip()

    if confirm == "y":
        dataset_name = os.path.basename(file_path)
        #logger.info(f"Menyimpan dataset ke folder lokal: {dataset_name}")
        base_name, ext = os.path.splitext(dataset_name)

        dest_root = ensure_data_dir()
        dataset_dir = os.path.join(dest_root, base_name)
        if folder_name:
            dataset_dir = os.path.join(dest_root, folder_name)
        
        os.makedirs(dataset_dir, exist_ok=True)

        if ext.lower() == ".zip":
            # Ekstrak ZIP ke subfolder dataset
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print(f" File ZIP diekstrak ke: {dataset_dir}")
        else:
            # Copy file biasa
            dest_path = os.path.join(dataset_dir, dataset_name)
            shutil.copy(file_path, dest_path)
            print(f" Dataset disalin ke: {dest_path}")

        #logger.info(f"Dataset disimpan di folder lokal: {dataset_dir}")
    else:
        print(" Dataset tidak disimpan (hanya preview sementara).")
def handle_remove_readonly(func, path, exc_info):
    """
    Ubah permission file agar bisa dihapus.
    Digunakan oleh shutil.rmtree untuk menangani file read-only (Windows & Mac/Linux).
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"⚠️ Tidak dapat menghapus: {path} ({e})")
def delete_local_dataset():
    """
    Menampilkan daftar folder dataset lokal, lalu memungkinkan pengguna menghapus satu folder.
    Aman digunakan di Windows dan macOS/Linux.
    """
    base_path = "data/local_datasets"
    if not os.path.exists(base_path):
        print("📁 Folder dataset lokal belum ada.")
        return

    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    if not folders:
        print("📭 Tidak ada folder dataset lokal untuk dihapus.")
        return

    print("\n=== Daftar Folder Dataset Lokal ===")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")

    # Tambahkan opsi hapus semua folder
    print(f"{len(folders) + 1}. Hapus semua folder dataset lokal ⚠️")

    try:
        idx = int(input("Masukkan nomor folder yang ingin dihapus: "))
        if idx < 1 or idx > len(folders) + 1:
            print("❌ Nomor tidak valid.")
            return

        # --- HAPUS SEMUA ---
        if idx == len(folders) + 1:
            konfirmasi = input("⚠️ Yakin ingin menghapus SEMUA dataset lokal? (y/n): ").lower()
            if konfirmasi == "y":
                for folder in folders:
                    folder_path = os.path.join(base_path, folder)
                    shutil.rmtree(folder_path, onerror=handle_remove_readonly)
                print("✅ Semua folder dataset lokal berhasil dihapus.")
            else:
                print("❎ Penghapusan dibatalkan.")
            return

        # --- HAPUS SATU FOLDER ---
        folder_to_delete = os.path.join(base_path, folders[idx - 1])
        konfirmasi = input(f"⚠️ Yakin ingin menghapus folder '{folders[idx - 1]}' beserta isinya? (y/n): ").lower()

        if konfirmasi == "y":
            shutil.rmtree(folder_to_delete, onerror=handle_remove_readonly)
            print(f"✅ Folder '{folders[idx - 1]}' berhasil dihapus.")
        else:
            print("❎ Penghapusan dibatalkan.")

    except ValueError:
        print("❌ Input tidak valid.")
    except PermissionError:
        print("🚫 Akses ditolak. Tutup aplikasi yang menggunakan file ini dan coba lagi.")
    except Exception as e:
        print(f"⚠️ Terjadi kesalahan: {e}")

            
#https://drive.google.com/file/d/1j5yCAUG5ooaOMFmrUjW0jNq90qQwrEjB/view?usp=sharing
# https://drive.google.com/uc?id=1j5yCAUG5ooaOMFmrUjW0jNq90qQwrEjB
# https://drive.google.com/file/d/1j5yCAUG5ooaOMFmrUjW0jNq90qQwrEjB/view?usp=sharing
#iannarsa/gofood-merchant-on-yogyakarta
# C:\Users\US-DIS~1\AppData\Local\Temp\gdrive_svm_thc9\gofood-merchant-on-yogyakarta.zip
