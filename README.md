# Membangun Model - Telco Customer Churn

Repository ini berisi eksperimen machine learning untuk memodelkan churn pelanggan Telco setelah proses preprocessing selesai. Di dalam folder ini ada dua alur training:

- `modelling.py` untuk training dasar menggunakan `RandomForestClassifier`.
- `modelling_tuning.py` untuk training lanjutan dengan `GridSearchCV`, logging metrik manual, dan penyimpanan artefak evaluasi.

## Isi Repository

```text
.
├── modelling.py
├── modelling_tuning.py
├── requirements.txt
├── DagsHub.txt
├── mlflow.db
├── mlruns/
├── artifacts/
└── telco_customer_churn_preprocessing/
	├── label_encoder.joblib
	├── preprocessor.joblib
	├── test_preprocessed.csv
	└── train_preprocessed.csv
```

## Deskripsi File dan Folder

### `modelling.py`

Script training dasar yang:

- membaca data train dan test dari folder `telco_customer_churn_preprocessing/`,
- memisahkan fitur dan target `Churn`,
- melatih `RandomForestClassifier`,
- menghitung akurasi dan menampilkan `classification_report`,
- menggunakan MLflow autologging.

### `modelling_tuning.py`

Script training lanjutan yang:

- memakai `RandomForestClassifier` dengan `GridSearchCV`,
- mencari parameter terbaik berdasarkan skor `f1`,
- menghitung metrik evaluasi seperti `accuracy`, `precision`, `recall`, `f1_score`, dan `roc_auc`,
- menyimpan artefak evaluasi ke folder `artifacts/`,
- logging model dan metrik ke MLflow,
- diarahkan ke DagsHub untuk experiment tracking.

### `telco_customer_churn_preprocessing/`

Folder berisi hasil preprocessing yang dipakai sebagai input training:

- `train_preprocessed.csv`
- `test_preprocessed.csv`
- `preprocessor.joblib`
- `label_encoder.joblib`

### `artifacts/`

Folder output dari proses tuning/eksperimen. Isi yang ada saat ini:

- `best_params.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `dataset_info.json`
- `feature_importance.csv`
- `metrics.json`
- `roc_curve.png`

### `mlflow.db` dan `mlruns/`

Artefak lokal untuk tracking MLflow. Folder dan database ini menyimpan histori run, eksperimen, dan metadata hasil training di mesin lokal.

### `DagsHub.txt`

Catatan konfigurasi tracking yang menunjuk ke:

- DagsHub repository: `https://dagshub.com/dickyhaa/MSML-Telco-Churn`
- MLflow experiment: `Telco Customer Churn Advanced Modelling`
- run name: `RandomForest_Tuning_Manual_Logging`

## Requirement

Dependensi Python yang tercantum di `requirements.txt`:

- `pandas`
- `numpy`
- `scikit-learn`
- `mlflow`
- `matplotlib`
- `joblib`
- `dagshub`

## Cara Menjalankan

### 1. Install dependensi

```bash
pip install -r requirements.txt
```

### 2. Jalankan training dasar

```bash
python modelling.py
```

### 3. Jalankan training lanjutan dengan tuning

```bash
python modelling_tuning.py
```

## Output yang Dihasilkan

Setelah menjalankan script tuning, repository ini dapat menghasilkan:

- metrik evaluasi model,
- classification report,
- confusion matrix,
- ROC curve,
- feature importance,
- parameter terbaik hasil tuning,
- logging eksperimen ke MLflow dan DagsHub.

## Catatan

- Data yang dipakai di sini sudah dalam bentuk preprocessed, jadi training tidak dimulai dari data mentah.
- File `.joblib` berfungsi sebagai artefak preprocessing/encoding yang siap dipakai ulang.
- Folder `mlruns/`, `mlflow.db`, dan `artifacts/` merupakan hasil eksperimen yang sudah ada di repository ini.
