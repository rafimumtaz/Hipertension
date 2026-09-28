########################################
Type: markdown
# Analisis Pengaruh Tingkat Konsumsi Garam terhadap Risiko Hipertensi Menggunakan Algoritma Klasifikasi LightGBM berbasis SHAP
########################################
Type: code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

########################################
Type: code
from google.colab import files

uploaded = files.upload()

nama_file = list(uploaded.keys())[0]
print('Nama file:', nama_file)

if nama_file.endswith('.csv'):
    df = pd.read_csv(nama_file)
elif nama_file.endswith(('.xlsx', '.xls')):
    df = pd.read_excel(nama_file)
########################################
Type: markdown
## membaca data
########################################
Type: code

display(df.head())

print('\nInfo Dataset:')
df.info()

print('\nStatistik Deskriptif:')
display(df.describe(include='all'))
########################################
Type: markdown
## distribusi variabel target
########################################
Type: code
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Has_Hypertension', palette='Set2')
plt.title('Distribusi Kelas Has_Hypertension')
plt.show()

# Tambahan: hitung angka pastinya
target_counts = df['Has_Hypertension'].value_counts()
target_pct = df['Has_Hypertension'].value_counts(normalize=True) * 100

print("Jumlah per kelas:")
print(target_counts)
print("\nPersentase per kelas:")
print(target_pct.round(2))

imbalance_ratio = target_counts.max() / target_counts.min()
print(f"\nRasio imbalance: {imbalance_ratio:.2f} : 1")
########################################
Type: markdown
## analisis numerik
########################################
Type: code
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Has_Hypertension', y='Salt_Intake', palette='Set2')
plt.title('Distribusi Konsumsi Garam Berdasarkan Status Hipertensi')
plt.show()
########################################
Type: markdown
## cek outlier
########################################
Type: code
# 1. Otomatis mendefinisikan kolom numerik (atau pilih manual)
num_cols_check = df.select_dtypes(include=['number']).columns.tolist()

# 2. Jalankan kode deteksi outlier Anda
print("=== Hasil Deteksi Outlier ===\n")
for col in num_cols_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    outlier = df[
        (df[col] < batas_bawah) |
        (df[col] > batas_atas)
    ]

    pct = len(outlier) / len(df) * 100

    print(f"{col}:")
    print(f"  Batas bawah : {batas_bawah:.2f}")
    print(f"  Batas atas  : {batas_atas:.2f}")
    print(f"  Jumlah outlier: {len(outlier)} ({pct:.2f}% dari total data)\n")
########################################
Type: markdown
## korelasi antar variabel numerik
########################################
Type: code
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Numerik')
plt.show()
########################################
Type: markdown
## fitur kategorial
########################################
Type: code
categorical_cols = df.select_dtypes(include=['object']).columns.drop('Has_Hypertension')
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=df, x=col, hue='Has_Hypertension', palette='Set2')
    plt.xticks(rotation=45)
    plt.title(f'{col} vs Hypertension')
plt.tight_layout()
plt.show()
########################################
Type: markdown
##  PLOT LENGKAP: AGREGASI, & ACUAN INDIKASI
########################################
Type: code

# ---------- PLOT 3: Agregasi Rata-rata Fitur Numerik (Yes vs No) ----------
agg_numeric = df.groupby('Has_Hypertension')[numerical_cols].mean().T

fig, ax = plt.subplots(figsize=(10, 6))
agg_numeric.plot(kind='bar', ax=ax, color=['#81b29a', '#e07a5f'])
ax.set_title('Agregasi: Rata-rata Fitur Numerik (Hipertensi vs Tidak)')
ax.set_ylabel('Nilai Rata-rata')
ax.set_xlabel('Fitur')
ax.tick_params(axis='x', rotation=30)
ax.legend(title='Has_Hypertension')

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)

plt.tight_layout()
plt.show()

# ---------- TABEL ACUAN GABUNGAN (Semua Fitur, Diurutkan Kekuatan Indikasi) ----------
acuan_rows = []

for col in categorical_cols:
    crosstab_pct = pd.crosstab(df[col], df['Has_Hypertension'], normalize='index') * 100
    for kategori in crosstab_pct.index:
        pct_yes = crosstab_pct.loc[kategori, 'Yes']
        status = "Terindikasi" if pct_yes > 50 else "Tidak Terindikasi"
        selisih = abs(pct_yes - 50)
        acuan_rows.append({
            'Label': f"{col}: {kategori}",
            'Nilai (%)': pct_yes,
            'Status': status,
            'Kekuatan Indikasi': round(selisih, 1)
        })

for col in numerical_cols:
    mean_yes = df[df['Has_Hypertension'] == 'Yes'][col].mean()
    mean_no = df[df['Has_Hypertension'] == 'No'][col].mean()
    threshold = (mean_yes + mean_no) / 2
    arah = ">" if mean_yes > mean_no else "<"
    selisih_relatif = abs(mean_yes - mean_no) / ((mean_yes + mean_no) / 2) * 100
    acuan_rows.append({
        'Label': f"{col} {arah} {threshold:.2f}",
        'Nilai (%)': selisih_relatif,
        'Status': "Terindikasi",
        'Kekuatan Indikasi': round(selisih_relatif, 1)
    })

df_acuan = pd.DataFrame(acuan_rows).sort_values('Kekuatan Indikasi', ascending=False).reset_index(drop=True)

print("\nTabel Acuan Indikasi Hipertensi (Semua Fitur/Kategori):")
display(df_acuan)
########################################
Type: code
print(df.dtypes)
########################################
Type: markdown
# preprocesing
########################################
Type: code
print("Ukuran dataset:", df.shape)
########################################
Type: markdown
## cek missing value
########################################
Type: code
missing = df.isnull().sum()
print('Missing Values:\n', missing)
########################################
Type: code
df['Medication'] = df['Medication'].astype('object')
df['Medication'] = df['Medication'].fillna('None')

print(df['Medication'].isnull().sum())
print(df['Medication'].value_counts())
########################################
Type: markdown
## encoding fitur kategorika;
########################################
Type: code
cat_cols = df.select_dtypes(include=['object']).columns
le_dict = {}

df_prepared = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_prepared[col] = le.fit_transform(df_prepared[col])

    df_prepared[col] = df_prepared[col].astype('category')
    le_dict[col] = le

print("=== TABEL 1: Data Hasil Encoding ===")
display(df_prepared.head())

#  Encoding Semua Fitur=
mapping_rows = []
for col, le in le_dict.items():
    for kategori, kode in zip(le.classes_, le.transform(le.classes_)):
        mapping_rows.append({'Fitur': col, 'Kategori Asli': kategori, 'Kode': kode})

df_mapping = pd.DataFrame(mapping_rows)

print("\n=== TABEL 2: Mapping Encoding Semua Fitur ===")
print(f"Jumlah baris tabel mapping: {len(df_mapping)}")
display(df_mapping)
########################################
Type: markdown
## menyimpan data prepared
########################################
Type: code

df_prepared.to_csv('prepared_dataset.csv', index=False)

print('Data berhasil disimpan sebagai prepared_dataset.csv')
########################################
Type: markdown
## menentukan fitur x dan y
########################################
Type: code
X = df_prepared.drop('Has_Hypertension', axis=1)
y = df_prepared['Has_Hypertension']
########################################
Type: markdown
## train-split data
########################################
Type: code

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
########################################
Type: markdown
# modeling
########################################
Type: code
df = pd.read_csv('prepared_dataset.csv')
cat_features = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
for col in cat_features:
    df[col] = df[col].astype('category')

X = df.drop('Has_Hypertension', axis=1)
y = df['Has_Hypertension']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
########################################
Type: markdown
##  training -lightgbm
########################################
Type: code
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)
########################################
Type: markdown
## evaluasi model
########################################
Type: code
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print('Classification Report:')
print(classification_report(y_test, y_pred))

print(f'ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
########################################
Type: markdown
## confusion matrix
########################################
Type: code
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
########################################
Type: markdown
## menyimpan model
########################################
Type: code
os.makedirs('../Models', exist_ok=True)
model_path = '../Models/lightgbm_model.pkl'
joblib.dump(model, model_path)
print(f'Model berhasil disimpan di: {model_path}')
########################################
Type: markdown
# evaluasi
########################################
Type: markdown
# Explainable AI (XAI) Menggunakan SHAP
########################################
Type: code
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values_plot = shap_values[1]
else:
    shap_values_plot = shap_values

plt.figure()
shap.summary_plot(shap_values_plot, X_test, show=False)
plt.title('SHAP Summary Plot - Pengaruh Fitur terhadap Prediksi Hipertensi')
plt.tight_layout()
plt.show()
########################################
Type: code
plt.figure()
shap.summary_plot(shap_values_plot, X_test, plot_type='bar', show=False)
plt.title('SHAP Feature Importance (Mean Absolute SHAP Value)')
plt.tight_layout()
plt.show()
########################################
Type: code
plt.figure()
shap.dependence_plot('Salt_Intake', shap_values_plot, X_test, show=False)
plt.title('SHAP Dependence Plot - Salt_Intake')
plt.tight_layout()
plt.show()
########################################
Type: markdown
## WATERFALL
########################################
Type: code
sample_idx = 0
explainer_exp = shap.Explainer(model)
shap_exp_values = explainer_exp(X_test)

plt.figure()
shap.plots.waterfall(shap_exp_values[sample_idx], show=False)
plt.title(f'SHAP Waterfall - Pasien index {sample_idx}')
plt.tight_layout()
plt.show()
########################################
Type: markdown
## RANKING FITUR PALING BERPENGARUH
########################################
Type: code
feature_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean_Abs_SHAP': np.abs(shap_values_plot).mean(axis=0)
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("=== RANKING FITUR BERDASARKAN SHAP ===")
print(feature_importance)
########################################
Type: code
