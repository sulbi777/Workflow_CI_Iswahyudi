"""
train_model.py
Melatih model prediksi produksi komoditas (ton) berdasarkan Provinsi + Komoditas,
menggunakan pipeline preprocessing yang sama seperti automate_Iswahyudi.py (Kriteria 1),
lalu men-tracking eksperimen dengan MLflow dan mengekspor artefak untuk serving (Kriteria 3).

Perbedaan kecil dari automate_Iswahyudi.py: di sini dipakai DUA LabelEncoder terpisah
(le_prov, le_komo) supaya encoder Provinsi tidak ikut ter-refit/ketimpa saat encode
Komoditas -- encoder ini WAJIB disimpan karena dipakai ulang saat inference
(mengubah input string "Provinsi"/"Komoditas" dari user menjadi kode integer yang
sama persis seperti saat training).
"""
import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

RAW_PATH = os.path.join(os.path.dirname(__file__), "data_clean.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def preprocess(input_path: str) -> pd.DataFrame:
    """Melt wide-format produksi per-komoditas menjadi long-format, lalu bersihkan."""
    df = pd.read_csv(input_path)
    prod_cols = [c for c in df.columns if "Produksi" in c and "(kuintal)" in c]
    df_melted = df.melt(
        id_vars=["Provinsi"], value_vars=prod_cols,
        var_name="Komoditas", value_name="Prod_Kw",
    )
    df_melted["Prod_Kw"] = df_melted["Prod_Kw"].replace("-", np.nan).astype(float)
    df_melted = df_melted.dropna().copy()
    df_melted["produksi_ton"] = df_melted["Prod_Kw"] / 10
    return df_melted


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("produksi-komoditas-msml")

    df = preprocess(RAW_PATH)

    # Encoder terpisah untuk tiap kolom kategorikal -> wajib untuk konsistensi saat serving
    le_prov = LabelEncoder()
    le_komo = LabelEncoder()
    df["prov_enc"] = le_prov.fit_transform(df["Provinsi"])
    df["komo_enc"] = le_komo.fit_transform(df["Komoditas"])

    X = df[["prov_enc", "komo_enc"]]
    y = df["produksi_ton"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = dict(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42)

    with mlflow.start_run(run_name="rf_produksi_ton"):
        model = RandomForestRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
            "r2": float(r2_score(y_test, y_pred)),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        print("Metrics:", json.dumps(metrics, indent=2))

    # Ekspor artefak untuk serving (Kriteria 3): model + kedua encoder + daftar kategori
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "model.pkl"))
    joblib.dump(le_prov, os.path.join(ARTIFACT_DIR, "le_prov.pkl"))
    joblib.dump(le_komo, os.path.join(ARTIFACT_DIR, "le_komo.pkl"))
    with open(os.path.join(ARTIFACT_DIR, "categories.json"), "w") as f:
        json.dump({
            "provinsi": sorted(le_prov.classes_.tolist()),
            "komoditas": sorted(le_komo.classes_.tolist()),
        }, f, ensure_ascii=False, indent=2)

    print(f"Artefak tersimpan di: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
