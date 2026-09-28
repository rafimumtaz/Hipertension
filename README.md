# Analisis Pengaruh Tingkat Konsumsi Garam terhadap Risiko Hipertensi Menggunakan Algoritma Klasifikasi LightGBM berbasis SHAP

## Project Overview
Proyek ini bertujuan untuk menganalisis dan memprediksi risiko hipertensi berdasarkan tingkat konsumsi garam dan faktor-faktor lainnya. Algoritma klasifikasi **LightGBM** digunakan karena efisiensinya, sementara **SHAP (SHapley Additive exPlanations)** digunakan untuk menjelaskan hasil model secara interpretatif.

## Metodologi (CRISP-DM)
Proyek ini mengikuti metodologi **CRISP-DM (Cross-Industry Standard Process for Data Mining)**:
1. **Business Understanding**: Memahami tujuan proyek dan mendefinisikan masalah.
2. **Data Understanding**: Eksplorasi data awal untuk memahami karakteristik dataset.
3. **Data Preparation**: Pembersihan data, transformasi, dan feature engineering.
4. **Modeling**: Membangun model prediksi menggunakan algoritma LightGBM.
5. **Evaluation**: Mengevaluasi model menggunakan metrik klasifikasi dan SHAP values.
6. **Deployment**: Menyiapkan model untuk digunakan dalam lingkungan produksi.

## Struktur Direktori
- `Datasets/`: Berisi dataset mentah dan yang sudah diproses.
- `Notebooks/`: Berisi Jupyter Notebook untuk setiap fase CRISP-DM.
- `src/`: Source code Python modular (jika diperlukan).
- `Models/`: Tempat menyimpan model machine learning yang sudah dilatih.
- `Reports/`: Hasil laporan, visualisasi, dan grafik.

## Cara Menjalankan Proyek (How to Run)

Ikuti langkah-langkah di bawah ini untuk mengatur dan menjalankan proyek dari awal:

### 1. Persiapan Environment (Virtual Environment)
Sangat disarankan menggunakan Virtual Environment untuk mengisolasi versi library yang digunakan.
- **Membuat Virtual Environment:**
  Buka terminal/command prompt di dalam folder proyek, lalu ketik:
  ```bash
  python -m venv venv
  ```
- **Mengaktifkan Virtual Environment:**
  - **Windows:** 
    ```bash
    .\venv\Scripts\activate
    ```
  - **Mac/Linux:** 
    ```bash
    source venv/bin/activate
    ```

### 2. Instalasi Dependencies
Setelah virtual environment aktif, instal semua library yang dibutuhkan (seperti pandas, scikit-learn, lightgbm, shap) menggunakan file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Membangun (Build) Notebooks
Proyek ini menggunakan script Python untuk men-generate struktur file Jupyter Notebooks secara otomatis agar sesuai dengan framework CRISP-DM. Jalankan perintah berikut untuk membuat semua file notebooks:
```bash
python build_notebooks.py
```
*Perintah ini akan membaca source code dan membuat file `.ipynb` dari tahap 01 hingga 05 di dalam folder `Notebooks/`. Selain itu, direktori seperti `Datasets/` dan `Models/` akan otomatis disiapkan jika belum ada.*

### 4. Menjalankan Jupyter Notebook
Setelah file notebooks berhasil di-generate, jalankan Jupyter Notebook untuk mengeksekusi analisis secara berurutan dan interaktif:
```bash
jupyter notebook
```
Buka URL yang muncul di browser Anda, navigasikan ke dalam folder `Notebooks/`, lalu buka dan *Run* (jalankan) file berikut secara berurutan:
1. `01_Business_Understanding.ipynb`
2. `02_Data_Understanding.ipynb`
3. `03_Data_Preparation.ipynb`
4. `04_Modeling.ipynb`
5. `05_Evaluation.ipynb`
