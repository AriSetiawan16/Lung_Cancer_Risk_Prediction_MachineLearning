# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1T3bcM2n4SvrJgzttb8-2VS6O4vythpeH

## Import Library
"""

# Visualisasi dan Analisis Data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Statistik & Preprocessing
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Model Selection & Evaluasi
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.datasets import make_classification

# Model Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Penanganan data imbalance
from imblearn.over_sampling import SMOTE

"""## Install Library"""

!pip install matplotlib seaborn pandas numpy scipy scikit-learn imbalanced-learn xgboost

"""## Load Dataset"""

df = pd.read_csv('/content/kesehatan.csv')

df.head()

"""## EDA"""

# Menampilkan informasi dasar tentang dataset
print("Informasi Data Frame:\n")
df_info = df.info()

# Menampilkan 5 baris terakhir untuk memastikan konsistensi data
print("5 Baris Terakhir dari Dataset:")
print(df.tail(), "\n")

# Menampilkan shape dataset (jumlah baris dan kolom)
print(f"Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom.\n")

# Menampilkan tipe data setiap kolom dalam dataset
print("Tipe Data Setiap Kolom:")
print(df.dtypes, "\n")

# Memeriksa jumlah nilai yang hilang (missing values) pada setiap kolom
print("Jumlah Missing Values per Kolom:")
print(df.isnull().sum(), "\n")

# Memeriksa dan menampilkan jumlah duplikasi data dalam dataset
print(f"Jumlah Data yang Duplikat: {df.duplicated().sum()}\n")

# Menampilkan statistik deskriptif dari dataset untuk mendapatkan gambaran umum dari distribusi data
print("Statistik Deskriptif Dataset:")
print(df.describe().transpose(), "\n")

# Ringkasan dalam bentuk tabel yang lebih terstruktur dengan format yang rapi
import tabulate
summary_table = {
    "Informasi Dataframe": [df_info],
    "5 Baris Terakhir": [df.tail()],
    "Jumlah Baris dan Kolom": [df.shape],
    "Tipe Data per Kolom": [df.dtypes],
    "Jumlah Missing Values": [df.isnull().sum()],
    "Jumlah Duplikasi": [df.duplicated().sum()],
    "Statistik Deskriptif": [df.describe().transpose()]
}

"""## Preprocessing Dataset"""

# Hapus duplikasi
df = df.drop_duplicates()

# Verifikasi ulang
print("Jumlah data duplikat setelah dibersihkan:", df.duplicated().sum())

# Membuat instance dari LabelEncoder
encoder = LabelEncoder()

# Mendefinisikan kolom-kolom yang memiliki data kategorikal
categorical_features = [
    'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
    'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE',
    'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING',
    'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY',
    'CHEST_PAIN', 'LUNG_CANCER'
]

# Menerapkan Label Encoding pada setiap kolom kategorikal
for feature in categorical_features:
    df[feature] = encoder.fit_transform(df[feature])

# Menampilkan beberapa baris untuk memverifikasi encoding
print(df.head())

# Menampilkan jumlah frekuensi masing-masing kategori pada kolom 'LUNG_CANCER'
lung_cancer_counts = df['LUNG_CANCER'].value_counts()
print("Distribusi Kelas Target LUNG_CANCER:")
print(lung_cancer_counts)

# Menggunakan hue untuk menghindari FutureWarning (assign x variable to hue)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='LUNG_CANCER', hue='LUNG_CANCER', palette='Set2', legend=False)
plt.title("Frekuensi Kelas Target Kanker Paru-paru")
plt.xlabel("Status Kanker Paru-paru")
plt.ylabel("Jumlah Kasus")
plt.tight_layout()  # Mengatur agar plot tidak saling bertumpuk
plt.show()

# Membuat pie chart untuk menunjukkan proporsi masing-masing kelas pada kolom 'LUNG_CANCER'
plt.figure(figsize=(6, 6))
plt.pie(lung_cancer_counts, labels=lung_cancer_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99'])
plt.title("Proporsi Kelas Target Kanker Paru-paru")
plt.tight_layout()  # Mengatur jarak agar tidak terlalu dekat
plt.show()

# Visualisasi distribusi target 'LUNG_CANCER' dengan countplot
plt.figure(figsize=(8, 6))
sns.countplot(x='LUNG_CANCER', data=df, palette='coolwarm')  # Menggunakan palette 'coolwarm' untuk warna yang berbeda
plt.title("Distribusi Status Kanker Paru-paru")
plt.xlabel("Status Kanker Paru-paru (Yes/No)")  # Menambahkan label untuk sumbu X
plt.ylabel("Jumlah Kasus")  # Menambahkan label untuk sumbu Y
plt.show()

# Heatmap Korelasi
numerical_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi')
plt.show()

"""## Modeling"""

# Membuat data simulasi
X, y = make_classification(n_samples=2998, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Total data: {len(X)}, Train: {len(X_train)}, Test: {len(X_test)}')

# Menstandarkan fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inisialisasi model yang digunakan
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Melatih model
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} telah dilatih.")

# Evaluasi model
results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)

    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Simpan hasil evaluasi
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": conf_matrix
    }

    print(f"{name} selesai dievaluasi dengan akurasi: {accuracy:.4f}")

# Tampilkan hasil evaluasi
for model, metrics in results.items():
    print(f"\nModel: {model}")
    for metric, value in metrics.items():
        if metric != "Confusion Matrix":
            print(f"  {metric}: {value:.4f}")
    print("-" * 50)

# Jumlah total model
model_names = list(results.keys())
n_models = len(model_names)

# Buat subplot: 3 kolom per baris (karena ada 3 model)
cols = 3
rows = (n_models + cols - 1) // cols  # pembulatan ke atas

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

# Flatten axes untuk akses mudah jika hanya 1 baris
axes = axes.flatten()

# Plot masing-masing confusion matrix
for idx, (name, metrics) in enumerate(results.items()):
    sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)),
                ax=axes[idx])
    axes[idx].set_title(f"Confusion Matrix - {name}")
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")

# Kosongkan subplot jika tidak terpakai
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Ambil nama model dan nilai akurasinya
accuracies = [metrics["Accuracy"] for metrics in results.values()]
accuracy_percent = [f"{acc * 100:.2f}%" for acc in accuracies]

# Buat plot perbandingan akurasi
plt.figure(figsize=(10, 6))
bars = sns.barplot(x=accuracies, y=model_names, palette="coolwarm")

# Tambahkan label persentase di ujung bar
for i, (bar, percent) in enumerate(zip(bars.patches, accuracy_percent)):
    plt.text(
        bar.get_width() + 0.01,  # Posisi x: sedikit di luar bar
        bar.get_y() + bar.get_height() / 2,  # Posisi y: tengah-tengah bar
        percent,
        va='center'
    )

# Pengaturan tambahan
plt.xlabel("Akurasi")
plt.title("Perbandingan Akurasi Model")
plt.xlim(0, 1.05)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""## Simulasi Prediksi Berdasarkan Data Sintetis"""

# Pisahkan fitur dan target dari dataset asli
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Split data menjadi data latih dan uji
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Total data: {len(X)}, Train: {len(X_train)}, Test: {len(X_test)}')

# Standarisasi data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Melatih model
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} telah dilatih.")

# Evaluasi model
results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)

    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Simpan hasil evaluasi
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": conf_matrix
    }

    print(f"{name} selesai dievaluasi dengan akurasi: {accuracy:.4f}")

# Tampilkan hasil evaluasi untuk model yang telah dilatih
for model, metrics in results.items():
    print(f"\nModel: {model}")
    for metric, value in metrics.items():
        if metric != "Confusion Matrix":
            print(f"  {metric}: {value:.4f}")
    print("-" * 50)

# Jumlah total model
model_names = list(results.keys())
n_models = len(model_names)

# Buat subplot untuk confusion matrix
cols = 3  # 3 model
rows = (n_models + cols - 1) // cols  # pembulatan ke atas

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

# Flatten axes untuk akses mudah jika hanya 1 baris
axes = axes.flatten()

# Plot masing-masing confusion matrix
for idx, (name, metrics) in enumerate(results.items()):
    sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)),
                ax=axes[idx])
    axes[idx].set_title(f"Confusion Matrix - {name}")
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")

# Kosongkan subplot jika tidak terpakai
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Ambil nama model dan nilai akurasinya
accuracies = [metrics["Accuracy"] for metrics in results.values()]
accuracy_percent = [f"{acc * 100:.2f}%" for acc in accuracies]

# Buat plot perbandingan akurasi
plt.figure(figsize=(10, 6))
bars = sns.barplot(x=accuracies, y=model_names, palette="coolwarm")

# Tambahkan label persentase di ujung bar
for i, (bar, percent) in enumerate(zip(bars.patches, accuracy_percent)):
    plt.text(
        bar.get_width() + 0.01,  # Posisi x: sedikit di luar bar
        bar.get_y() + bar.get_height() / 2,  # Posisi y: tengah-tengah bar
        percent,
        va='center'
    )

# Pengaturan tambahan
plt.xlabel("Akurasi")
plt.title("Perbandingan Akurasi Model (dalam Persentase Data Klinis)")
plt.xlim(0, 1.05)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""## Pemodelan

Tahap **Pemodelan** merupakan inti dari proses prediksi, di mana berbagai algoritma pembelajaran mesin diterapkan untuk memahami pola dalam data pelatihan dan kemudian digunakan untuk memprediksi hasil pada data pengujian. Dalam penelitian ini, pemodelan dilakukan menggunakan tiga pendekatan dataset yang berbeda: **Data Simulasi** dan **Data Klinis**, untuk membandingkan kinerja model dalam dua skenario yang berbeda.

### 1. Logistic Regression

**Deskripsi:**  
Logistic Regression adalah model klasifikasi biner yang memprediksi probabilitas suatu kejadian berdasarkan fungsi logistik. Model ini efektif untuk dataset dengan ukuran kecil hingga menengah, dan bekerja dengan baik jika terdapat hubungan linier antara fitur dan target.

**Parameter:**
- `penalty`: Regularisasi L2 (default)
- `C=1.0`: Parameter regulasi (semakin kecil, semakin kuat regulasi)
- `solver='lbfgs'`
- `max_iter=10000`: Memastikan model memiliki cukup iterasi untuk konvergensi

**Alasan Pemilihan:**  
Logistic Regression digunakan sebagai model dasar karena kesederhanaannya dan kemudahan interpretasinya. Model ini memberikan baseline yang baik untuk membandingkan model yang lebih kompleks.

**Performa:**
- **Data Simulasi**: Akurasi 97.00%, Precision 97.00%, Recall 97.00%, F1-Score 97.00%
- **Data Klinis**: Akurasi 49.67%, Precision 49.66%, Recall 49.67%, F1-Score 49.56%

---

### 2. Random Forest

**Deskripsi:**  
Random Forest adalah model ansambel yang terdiri dari banyak pohon keputusan, yang masing-masing dilatih pada subset data dan fitur yang berbeda. Hasil prediksi akhir diperoleh dengan merata-ratakan hasil dari semua pohon keputusan. Metode ini mengurangi overfitting dengan teknik bagging dan mampu menangani data yang lebih beragam.

**Parameter:**
- `n_estimators=100`: Jumlah pohon dalam hutan
- `max_features='auto'`: Memilih fitur secara acak untuk setiap pohon
- `random_state=42`: Untuk memastikan reprodusibilitas hasil

**Alasan Pemilihan:**  
Model ini lebih stabil dan tahan terhadap overfitting. Random Forest sangat efektif untuk dataset yang lebih kompleks dan dapat menangani noise pada data klinis yang lebih beragam.

**Performa:**
- **Data Simulasi**: Akurasi 98.17%, Precision 98.18%, Recall 98.17%, F1-Score 98.17%
- **Data Klinis**: Akurasi 53.67%, Precision 53.67%, Recall 53.67%, F1-Score 53.65%

---

### 3. XGBoost

**Deskripsi:**  
XGBoost adalah implementasi efisien dari gradient boosting yang menambahkan regularisasi untuk mengurangi overfitting. Model ini sering digunakan dalam kompetisi pembelajaran mesin karena akurasi yang tinggi dan efisiensi komputasi yang baik.

**Parameter:**
- `use_label_encoder=False`
- `eval_metric='logloss'`
- `n_estimators=100`: Jumlah model berurutan
- `learning_rate=0.1`: Mengontrol kontribusi setiap model terhadap prediksi akhir
- `random_state=42`: Untuk memastikan reprodusibilitas hasil

**Alasan Pemilihan:**  
XGBoost dipilih karena kemampuannya dalam menangani data besar dan fitur yang banyak, serta memiliki performa yang sangat baik dalam banyak aplikasi praktis.

**Performa:**
- **Data Simulasi**: Akurasi 97.83%, Precision 97.84%, Recall 97.83%, F1-Score 97.83%
- **Data Klinis**: Akurasi 51.17%, Precision 51.17%, Recall 51.17%, F1-Score 51.12%

---

## Perbandingan Performa Model

Terdapat perbedaan signifikan pada performa model antara data simulasi dan data klinis nyata:

#### Data Simulasi
Data simulasi dibuat dengan fungsi `make_classification()` dari scikit-learn yang menghasilkan data sintetis dengan pola yang mudah dikenali:

```python
X, y = make_classification(n_samples=2998, n_features=5, random_state=42)
```

Semua model menunjukkan performa sangat tinggi (96.83% - 98.17%), dengan Random Forest sebagai model terbaik.

#### Data Klinis
Data klinis nyata berasal dari dataset kanker paru-paru dengan variabel target 'LUNG_CANCER':

```python
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']
```

Performa pada data klinis jauh lebih rendah (47.00% - 55.50%), menunjukkan kompleksitas dan tantangan pada data medis nyata dibandingkan data simulasi.



### Perbandingan Performa Model

Performa pada data klinis jauh lebih rendah (47.00% - 55.50%), menunjukkan kompleksitas dan tantangan pada data medis nyata dibandingkan data simulasi.

| Model               | Data Simulasi    |               |               |               | Data Klinis     |               |               |               |
|---------------------|------------------|---------------|---------------|---------------|-----------------|---------------|---------------|---------------|
|                     | **Accuracy**      | **Precision** | **Recall**     | **F1-Score**  | **Accuracy**     | **Precision**  | **Recall**     | **F1-Score**  |
| Logistic Regression | 0.9700           | 0.9700        | 0.9700        | 0.9700        | 0.4967          | 0.4966         | 0.4967         | 0.4956         |
| Random Forest       | **0.9817**       | **0.9818**    | **0.9817**    | **0.9817**    | 0.5367          | 0.5367         | 0.5367         | 0.5365         |
| XGBoost             | 0.9783           | 0.9784        | 0.9783        | 0.9783        | 0.5117          | 0.5117         | 0.5117         | 0.5112         |


"""