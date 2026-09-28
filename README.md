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

## Instalasi
1. Aktifkan virtual environment:
   - Windows: `.\venv\Scripts\activate`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Jalankan Jupyter Notebook:
   `jupyter notebook`
