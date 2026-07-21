# Workflow-CI

Repository kriteria 3 target Skilled.

Workflow GitHub Actions berada di `.github/workflows/mlflow-ci.yml` dan menjalankan:

```bash
mlflow run MLProject -P data_path=students_dropout_preprocessing/students_dropout_preprocessed.csv
```

Setelah workflow berhasil, artifact `mlflow-training-artifacts` akan tersedia pada halaman run GitHub Actions.
