import nbformat
import os

def create_nb_01():
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell("# 1. Business Understanding\n\n## 1.1 Latar Belakang\nHipertensi atau tekanan darah tinggi adalah salah satu faktor risiko utama untuk berbagai penyakit kardiovaskular. Salah satu faktor gaya hidup yang paling sering dikaitkan dengan hipertensi adalah tingkat konsumsi garam (natrium). \n\nPenelitian ini bertujuan untuk melakukan analisis mendalam mengenai pengaruh tingkat konsumsi garam dan faktor-faktor lain (seperti usia, tingkat stres, riwayat hipertensi, BMI, dll) terhadap risiko terjadinya hipertensi.\n\n## 1.2 Tujuan Proyek\n1. Memahami karakteristik data dan distribusi setiap fitur.\n2. Mengidentifikasi hubungan antara tingkat konsumsi garam dengan kejadian hipertensi.\n3. Membangun model klasifikasi menggunakan algoritma **LightGBM** untuk memprediksi risiko hipertensi.\n4. (Fase Selanjutnya) Menginterpretasikan model menggunakan **SHAP** untuk memahami kontribusi setiap fitur terhadap prediksi model.\n\n## 1.3 Deskripsi Fitur\n- `Age`: Usia pasien (Tahun).\n- `Salt_Intake`: Estimasi asupan garam harian.\n- `Stress_Score`: Skor tingkat stres pasien (0-10).\n- `BP_History`: Riwayat tekanan darah (Normal, Prehypertension, Hypertension).\n- `Sleep_Duration`: Durasi tidur harian (Jam).\n- `BMI`: Body Mass Index.\n- `Medication`: Jenis obat yang sedang dikonsumsi.\n- `Family_History`: Riwayat hipertensi di keluarga (Yes/No).\n- `Exercise_Level`: Tingkat aktivitas fisik (Low, Moderate, High).\n- `Smoking_Status`: Status merokok (Smoker, Non-Smoker).\n- `Has_Hypertension`: Variabel Target (Yes/No)."))
    return nb

def create_nb_02():
    nb = nbformat.v4.new_notebook()
    nb.cells.extend([
        nbformat.v4.new_markdown_cell("# 2. Data Understanding (EDA)"),
        nbformat.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Set plotting style\nsns.set_theme(style='whitegrid')"),
        nbformat.v4.new_markdown_cell("## 2.1 Load Dataset"),
        nbformat.v4.new_code_cell("df = pd.read_csv('../Datasets/hypertension_dataset.csv')\ndisplay(df.head())\nprint('\\nInfo Dataset:')\ndf.info()\nprint('\\nStatistik Deskriptif:')\ndisplay(df.describe(include='all'))"),
        nbformat.v4.new_markdown_cell("## 2.2 Distribusi Variabel Target (Has_Hypertension)"),
        nbformat.v4.new_code_cell("plt.figure(figsize=(6, 4))\nsns.countplot(data=df, x='Has_Hypertension', palette='Set2')\nplt.title('Distribusi Kelas Has_Hypertension')\nplt.show()\n\ntarget_counts = df['Has_Hypertension'].value_counts()\ntarget_pct = df['Has_Hypertension'].value_counts(normalize=True) * 100\n\nprint(\"Jumlah per kelas:\")\nprint(target_counts)\nprint(\"\\nPersentase per kelas:\")\nprint(target_pct.round(2))\n\nimbalance_ratio = target_counts.max() / target_counts.min()\nprint(f\"\\nRasio imbalance: {imbalance_ratio:.2f} : 1\")"),
        nbformat.v4.new_markdown_cell("## 2.3 Analisis Numerik: Salt Intake vs Hypertension"),
        nbformat.v4.new_code_cell("plt.figure(figsize=(8, 5))\nsns.boxplot(data=df, x='Has_Hypertension', y='Salt_Intake', palette='Set2')\nplt.title('Distribusi Konsumsi Garam Berdasarkan Status Hipertensi')\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 2.4 Cek Outlier"),
        nbformat.v4.new_code_cell("num_cols_check = df.select_dtypes(include=['number']).columns.tolist()\n\nprint(\"=== Hasil Deteksi Outlier ===\\n\")\nfor col in num_cols_check:\n    Q1 = df[col].quantile(0.25)\n    Q3 = df[col].quantile(0.75)\n    IQR = Q3 - Q1\n    batas_bawah = Q1 - 1.5 * IQR\n    batas_atas = Q3 + 1.5 * IQR\n\n    outlier = df[(df[col] < batas_bawah) | (df[col] > batas_atas)]\n    pct = len(outlier) / len(df) * 100\n\n    print(f\"{col}:\")\n    print(f\"  Batas bawah : {batas_bawah:.2f}\")\n    print(f\"  Batas atas  : {batas_atas:.2f}\")\n    print(f\"  Jumlah outlier: {len(outlier)} ({pct:.2f}% dari total data)\\n\")"),
        nbformat.v4.new_markdown_cell("## 2.5 Korelasi Antar Variabel Numerik"),
        nbformat.v4.new_code_cell("numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns\ncorr = df[numeric_cols].corr()\nplt.figure(figsize=(8, 6))\nsns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Heatmap Korelasi Numerik')\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 2.6 Distribusi Fitur Kategorikal"),
        nbformat.v4.new_code_cell("categorical_cols = df.select_dtypes(include=['object']).columns.drop('Has_Hypertension')\nplt.figure(figsize=(15, 10))\nfor i, col in enumerate(categorical_cols, 1):\n    plt.subplot(2, 3, i)\n    sns.countplot(data=df, x=col, hue='Has_Hypertension', palette='Set2')\n    plt.xticks(rotation=45)\n    plt.title(f'{col} vs Hypertension')\nplt.tight_layout()\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 2.7 Tabel Acuan Indikasi Hipertensi"),
        nbformat.v4.new_code_cell("# ---------- PLOT Agregasi Rata-rata Fitur Numerik ----------\nagg_numeric = df.groupby('Has_Hypertension')[numeric_cols].mean().T\n\nfig, ax = plt.subplots(figsize=(10, 6))\nagg_numeric.plot(kind='bar', ax=ax, color=['#81b29a', '#e07a5f'])\nax.set_title('Agregasi: Rata-rata Fitur Numerik (Hipertensi vs Tidak)')\nax.set_ylabel('Nilai Rata-rata')\nax.set_xlabel('Fitur')\nax.tick_params(axis='x', rotation=30)\nax.legend(title='Has_Hypertension')\n\nfor container in ax.containers:\n    ax.bar_label(container, fmt='%.2f', padding=3)\n\nplt.tight_layout()\nplt.show()\n\n# ---------- TABEL ACUAN GABUNGAN ----------\nacuan_rows = []\n\nfor col in categorical_cols:\n    crosstab_pct = pd.crosstab(df[col], df['Has_Hypertension'], normalize='index') * 100\n    for kategori in crosstab_pct.index:\n        pct_yes = crosstab_pct.loc[kategori, 'Yes']\n        status = \"Terindikasi\" if pct_yes > 50 else \"Tidak Terindikasi\"\n        selisih = abs(pct_yes - 50)\n        acuan_rows.append({\n            'Label': f\"{col}: {kategori}\",\n            'Nilai (%)': pct_yes,\n            'Status': status,\n            'Kekuatan Indikasi': round(selisih, 1)\n        })\n\nfor col in numeric_cols:\n    mean_yes = df[df['Has_Hypertension'] == 'Yes'][col].mean()\n    mean_no = df[df['Has_Hypertension'] == 'No'][col].mean()\n    threshold = (mean_yes + mean_no) / 2\n    arah = \">\" if mean_yes > mean_no else \"<\"\n    selisih_relatif = abs(mean_yes - mean_no) / ((mean_yes + mean_no) / 2) * 100\n    acuan_rows.append({\n        'Label': f\"{col} {arah} {threshold:.2f}\",\n        'Nilai (%)': selisih_relatif,\n        'Status': \"Terindikasi\",\n        'Kekuatan Indikasi': round(selisih_relatif, 1)\n    })\n\ndf_acuan = pd.DataFrame(acuan_rows).sort_values('Kekuatan Indikasi', ascending=False).reset_index(drop=True)\n\nprint(\"\\nTabel Acuan Indikasi Hipertensi (Semua Fitur/Kategori):\")\ndisplay(df_acuan)")
    ])
    return nb

def create_nb_03():
    nb = nbformat.v4.new_notebook()
    nb.cells.extend([
        nbformat.v4.new_markdown_cell("# 3. Data Preparation"),
        nbformat.v4.new_code_cell("import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\nimport os"),
        nbformat.v4.new_markdown_cell("## 3.1 Load Dataset"),
        nbformat.v4.new_code_cell("df = pd.read_csv('../Datasets/hypertension_dataset.csv')\nprint(f'Total Data Awal: {df.shape}')"),
        nbformat.v4.new_markdown_cell("## 3.2 Handling Missing Values"),
        nbformat.v4.new_code_cell("missing = df.isnull().sum()\nprint('Missing Values Awal:\\n', missing)\n\n# Mengisi missing value pada Medication\ndf['Medication'] = df['Medication'].astype('object')\ndf['Medication'] = df['Medication'].fillna('None')\n\nprint('\\nMissing values pada Medication setelah diisi:', df['Medication'].isnull().sum())\nprint('\\nDistribusi nilai Medication:')\nprint(df['Medication'].value_counts())"),
        nbformat.v4.new_markdown_cell("## 3.3 Encoding Fitur Kategorikal\nKarena LightGBM dapat menangani data kategorikal secara native jika diubah menjadi integer tipe kategori, kita akan menggunakan `LabelEncoder` untuk seluruh kolom objek (termasuk target)."),
        nbformat.v4.new_code_cell("cat_cols = df.select_dtypes(include=['object', 'str']).columns\nle_dict = {}\n\ndf_prepared = df.copy()\nfor col in cat_cols:\n    le = LabelEncoder()\n    encoded_values = le.fit_transform(df_prepared[col].astype(str))\n    df_prepared[col] = pd.Series(encoded_values, index=df_prepared.index).astype('category')\n    le_dict[col] = le\n\nprint(\"=== TABEL 1: Data Hasil Encoding ===\")\ndisplay(df_prepared.head())\n\nmapping_rows = []\nfor col, le in le_dict.items():\n    for kategori, kode in zip(le.classes_, le.transform(le.classes_)):\n        mapping_rows.append({'Fitur': col, 'Kategori Asli': kategori, 'Kode': kode})\n\ndf_mapping = pd.DataFrame(mapping_rows)\n\nprint(\"\\n=== TABEL 2: Mapping Encoding Semua Fitur ===\")\nprint(f\"Jumlah baris tabel mapping: {len(df_mapping)}\")\ndisplay(df_mapping)"),
        nbformat.v4.new_markdown_cell("## 3.4 Menyimpan Data Prepared"),
        nbformat.v4.new_code_cell("os.makedirs('../Datasets', exist_ok=True)\ndf_prepared.to_csv('../Datasets/prepared_dataset.csv', index=False)\nprint('Data berhasil disimpan ke ../Datasets/prepared_dataset.csv')"),
        nbformat.v4.new_markdown_cell("## 3.5 Contoh Train-Test Split\nSebagai ilustrasi awal, kita akan membagi dataset menjadi 3 skenario berbeda: 70-30, 80-20, dan 90-10. Nantinya, pada tahapan *Modeling*, kita akan membandingkan performa model pada ketiga split ini."),
        nbformat.v4.new_code_cell("X = df_prepared.drop('Has_Hypertension', axis=1)\ny = df_prepared['Has_Hypertension']\n\n# Split 70-30\nX_train_70, X_test_70, y_train_70, y_test_70 = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\nprint('Skenario 70-30 -> X_train shape:', X_train_70.shape, '| X_test shape:', X_test_70.shape)\n\n# Split 80-20\nX_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\nprint('Skenario 80-20 -> X_train shape:', X_train_80.shape, '| X_test shape:', X_test_80.shape)\n\n# Split 90-10\nX_train_90, X_test_90, y_train_90, y_test_90 = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)\nprint('Skenario 90-10 -> X_train shape:', X_train_90.shape, '| X_test shape:', X_test_90.shape)")
    ])
    return nb

def create_nb_04():
    nb = nbformat.v4.new_notebook()
    nb.cells.extend([
        nbformat.v4.new_markdown_cell("# 4. Modeling (LightGBM)"),
        nbformat.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport lightgbm as lgb\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport joblib\nimport os\nimport time\nimport warnings\nwarnings.filterwarnings('ignore')"),
        nbformat.v4.new_markdown_cell("## 4.1 Load Prepared Dataset"),
        nbformat.v4.new_code_cell("df = pd.read_csv('../Datasets/prepared_dataset.csv')\n\n# Convert categorical columns to 'category' type\ncat_features = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']\nfor col in cat_features:\n    df[col] = df[col].astype('category')\n\nX = df.drop('Has_Hypertension', axis=1)\ny = df['Has_Hypertension']"),
        nbformat.v4.new_markdown_cell("## 4.2 LightGBM Training & Split Comparison\nKita akan membandingkan 3 skenario split data: 70-30, 80-20, dan 90-10, lalu mengukur waktu pelatihan serta akurasinya."),
        nbformat.v4.new_code_cell("splits = [0.3, 0.2, 0.1]\nsplit_names = ['70-30', '80-20', '90-10']\nresults = []\nmodels = {}\n\nfor test_size, name in zip(splits, split_names):\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)\n    \n    model = lgb.LGBMClassifier(random_state=42)\n    \n    start_time = time.time()\n    model.fit(X_train, y_train)\n    end_time = time.time()\n    \n    training_time = end_time - start_time\n    \n    y_pred = model.predict(X_test)\n    accuracy = accuracy_score(y_test, y_pred)\n    \n    results.append({\n        'Split': name,\n        'Training Time (s)': training_time,\n        'Accuracy': accuracy\n    })\n    models[name] = {'model': model, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}\n\ndf_results = pd.DataFrame(results)\nprint(\"=== Perbandingan Skenario Split ===\")\ndisplay(df_results)"),
        nbformat.v4.new_markdown_cell("## 4.3 Evaluasi Model Terbaik (Skenario 80-20)\nUntuk evaluasi dan tahapan SHAP lebih lanjut, kita simpan dan gunakan skenario standar 80-20."),
        nbformat.v4.new_code_cell("best_model_info = models['80-20']\nbest_model = best_model_info['model']\nX_test_best = best_model_info['X_test']\ny_test_best = best_model_info['y_test']\ny_pred_best = best_model_info['y_pred']\ny_proba_best = best_model.predict_proba(X_test_best)[:, 1]\n\nprint('Classification Report (80-20):')\nprint(classification_report(y_test_best, y_pred_best))\n\nprint(f'ROC-AUC Score: {roc_auc_score(y_test_best, y_proba_best):.4f}')\nprint(f'Accuracy: {accuracy_score(y_test_best, y_pred_best):.4f}')"),
        nbformat.v4.new_markdown_cell("### 4.3.1 Confusion Matrix (80-20)"),
        nbformat.v4.new_code_cell("cm = confusion_matrix(y_test_best, y_pred_best)\nplt.figure(figsize=(5,4))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\nplt.title('Confusion Matrix (80-20)')\nplt.ylabel('Actual')\nplt.xlabel('Predicted')\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 4.4 Menyimpan Model"),
        nbformat.v4.new_code_cell("os.makedirs('../Models', exist_ok=True)\nmodel_path = '../Models/lightgbm_model.pkl'\njoblib.dump(best_model, model_path)\nprint(f'Model 80-20 berhasil disimpan di: {model_path}')")
    ])
    return nb

def create_nb_05():
    nb = nbformat.v4.new_notebook()
    nb.cells.extend([
        nbformat.v4.new_markdown_cell("# 5. Evaluation (Explainable AI with SHAP)"),
        nbformat.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport lightgbm as lgb\nimport shap\nimport joblib\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\nsns.set_theme(style='whitegrid')\nfrom sklearn.model_selection import train_test_split"),
        nbformat.v4.new_markdown_cell("## 5.1 Load Data dan Model"),
        nbformat.v4.new_code_cell("df = pd.read_csv('../Datasets/prepared_dataset.csv')\n\n# Convert categorical columns\ncat_features = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']\nfor col in cat_features:\n    df[col] = df[col].astype('category')\n\nX = df.drop('Has_Hypertension', axis=1)\ny = df['Has_Hypertension']\n\n# Gunakan random_state yang sama untuk mendapatkan X_test yang persis sama\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\n# Load Model\nmodel = joblib.load('../Models/lightgbm_model.pkl')\nprint('Model LightGBM dan data uji berhasil dimuat.')"),
        nbformat.v4.new_markdown_cell("## 5.2 SHAP Explainer"),
        nbformat.v4.new_code_cell("explainer = shap.TreeExplainer(model)\nshap_values = explainer.shap_values(X_test)\n\nif isinstance(shap_values, list):\n    shap_values_plot = shap_values[1]\nelse:\n    shap_values_plot = shap_values"),
        nbformat.v4.new_markdown_cell("## 5.3 SHAP Summary Plot"),
        nbformat.v4.new_code_cell("plt.figure()\nshap.summary_plot(shap_values_plot, X_test, show=False)\nplt.title('SHAP Summary Plot - Pengaruh Fitur terhadap Prediksi Hipertensi')\nplt.tight_layout()\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 5.4 SHAP Feature Importance"),
        nbformat.v4.new_code_cell("plt.figure()\nshap.summary_plot(shap_values_plot, X_test, plot_type='bar', show=False)\nplt.title('SHAP Feature Importance (Mean Absolute SHAP Value)')\nplt.tight_layout()\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 5.5 SHAP Dependence Plot"),
        nbformat.v4.new_code_cell("plt.figure()\nshap.dependence_plot('Salt_Intake', shap_values_plot, X_test, show=False)\nplt.title('SHAP Dependence Plot - Salt_Intake')\nplt.tight_layout()\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 5.6 SHAP Waterfall Plot"),
        nbformat.v4.new_code_cell("sample_idx = 0\nexplainer_exp = shap.Explainer(model)\nshap_exp_values = explainer_exp(X_test)\n\nplt.figure()\nshap.plots.waterfall(shap_exp_values[sample_idx], show=False)\nplt.title(f'SHAP Waterfall - Pasien index {sample_idx}')\nplt.tight_layout()\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 5.7 Ranking Fitur Paling Berpengaruh"),
        nbformat.v4.new_code_cell("feature_importance = pd.DataFrame({\n    'Feature': X_test.columns,\n    'Mean_Abs_SHAP': np.abs(shap_values_plot).mean(axis=0)\n}).sort_values('Mean_Abs_SHAP', ascending=False)\n\nprint(\"=== RANKING FITUR BERDASARKAN SHAP ===\")\nprint(feature_importance)")
    ])
    return nb

def main():
    notebooks = {
        'Notebooks/01_Business_Understanding.ipynb': create_nb_01(),
        'Notebooks/02_Data_Understanding.ipynb': create_nb_02(),
        'Notebooks/03_Data_Preparation.ipynb': create_nb_03(),
        'Notebooks/04_Modeling.ipynb': create_nb_04(),
        'Notebooks/05_Evaluation.ipynb': create_nb_05()
    }
    for path, nb in notebooks.items():
        # Make sure directory exists just in case
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Created {path}")

if __name__ == '__main__':
    main()
