# Monitoring & Logging (MSML Eksperimen Dataset)

Sistem monitoring untuk model prediksi produksi komoditas (ton) berdasarkan
Provinsi + Komoditas, dibangun di atas pipeline preprocessing `automate_Iswahyudi.py`
(Kriteria 1).

## Struktur proyek

```
kriteria3/
├── model/
│   ├── automate_Iswahyudi.py   # preprocessing asli (Kriteria 1)
│   ├── data_clean.csv          # dataset mentah (wide-format per provinsi/komoditas)
│   ├── train_model.py          # preprocessing + training + MLflow tracking
│   └── artifacts/              # output training (model, encoder, kategori)
├── serving/
│   ├── inference.py            # Flask API: /predict, /health, /metrics
│   ├── requirements.txt
│   ├── Dockerfile
│   └── artifacts/              # salinan model utk di-build ke image
├── monitoring/
│   ├── prometheus.yml          # scrape config
│   ├── alert_rules.yml         # 5 alert rules (syarat advanced: >=3)
│   └── alertmanager.yml        # routing notifikasi (webhook Discord/Slack)
├── grafana/
│   ├── provisioning/           # auto-provision datasource + dashboard
│   └── dashboards/
│       └── ml_monitoring_dashboard.json   # 12 panel (syarat advanced: >=10 metrics)
├── docker-compose.yml          # orkestrasi: inference + prometheus + alertmanager + grafana
└── .github/workflows/cd-docker.yml   # build & push image ke Docker Hub via CI
```

## Catatan penting soal model

Belum ada model/artefak Kriteria 1 yang di-share, jadi `train_model.py` di sini
melatih **RandomForestRegressor** baseline sendiri (fitur: `prov_enc`, `komo_enc`
→ target: `produksi_ton`), dengan MLflow tracking (`sqlite:///mlflow.db`).
Hasil sudah dites: MAE ±22.6rb ton, R² 0.40 — wajar untuk 2 fitur kategorikal saja,
dan bukan fokus kriteria ini. **Kalau model asli Kriteria 1/2 Anda berbeda**
(fitur lain, algoritma lain), ganti isi `model/train_model.py` dan jalankan ulang —
`serving/inference.py` hanya butuh 3 artefak: `model.pkl`, `le_prov.pkl`, `le_komo.pkl`.

Perbedaan kecil dari `automate_Iswahyudi.py` asli: di sini dipakai **dua** `LabelEncoder`
terpisah (satu utk Provinsi, satu utk Komoditas) alih-alih satu objek yang di-refit dua
kali — supaya encoder Provinsi tidak ketimpa mapping Komoditas saat disimpan untuk serving.

## Menjalankan lokal

```bash
docker compose up --build -d
```

- Inference API: http://localhost:8000 (`POST /predict`, `GET /metrics`, `GET /health`)
- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093
- Grafana: http://localhost:3000 (login default: `admin` / `admin`, dashboard sudah otomatis ter-load)

Contoh request prediksi:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"provinsi": "Aceh", "komoditas": "Produksi Bawang Daun (kuintal) (Kw)"}'
```

Daftar nama provinsi/komoditas yang valid ada di `model/artifacts/categories.json`.

## Setup notifikasi Alertmanager

Edit `monitoring/alertmanager.yml`, ganti URL webhook placeholder dengan webhook
Discord/Slack Anda sendiri (jangan commit URL asli ke repo publik — pertimbangkan
pakai secret/env var kalau repo publik).

## Setup CI: push image ke Docker Hub

Workflow `.github/workflows/cd-docker.yml` **tidak menyimpan password apa pun** —
ia baca dari GitHub Secrets. Yang perlu Anda lakukan di GitHub:

1. Buka repo → **Settings → Secrets and variables → Actions → New repository secret**
2. Tambahkan dua secret:
   - `DOCKERHUB_USERNAME` → username Docker Hub Anda
   - `DOCKERHUB_TOKEN` → **Access Token** (bukan password akun!), dibuat di
     Docker Hub: **Account Settings → Security → New Access Token**
3. Push ke branch `main` (atau jalankan manual lewat tab Actions → Run workflow)

Image akan otomatis ter-push ke `docker.io/<DOCKERHUB_USERNAME>/msml-inference:latest`.


