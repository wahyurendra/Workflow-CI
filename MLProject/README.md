# Workflow CI MLProject

Folder ini dipakai oleh GitHub Actions untuk menjalankan retraining model dengan MLflow Project.

Perintah lokal:

```bash
mlflow run . -P data_path=winequality_preprocessing/winequality_preprocessed.csv
```

Artifact model akan tersimpan pada folder `mlruns` dan diunggah oleh workflow sebagai GitHub Actions artifact.
