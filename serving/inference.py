"""
inference.py
Flask serving API untuk model prediksi produksi komoditas (Kriteria 3 Advanced).

Endpoint:
  POST /predict   -> {"provinsi": "...", "komoditas": "..."} => prediksi produksi_ton
  GET  /health     -> health check sederhana
  GET  /metrics    -> exposition Prometheus (scrape target)

Total 11 custom metrics diekspos (memenuhi syarat >=10 untuk level Advanced):
  1. ml_prediction_requests_total          (Counter, label status=success/error)
  2. ml_prediction_latency_seconds         (Histogram)
  3. ml_predicted_value_ton                (Histogram)  -> distribusi nilai prediksi
  4. ml_requests_in_progress               (Gauge)
  5. ml_validation_errors_total            (Counter)    -> provinsi/komoditas tak dikenal
  6. ml_last_prediction_timestamp_seconds  (Gauge)
  7. ml_cpu_usage_percent                  (Gauge)
  8. ml_memory_usage_percent               (Gauge)
  9. ml_memory_usage_bytes                 (Gauge)
  10. ml_disk_usage_percent                (Gauge)
  11. ml_app_uptime_seconds                (Gauge)
"""
import os
import time
import logging

import joblib
import psutil
from flask import Flask, request, jsonify, Response
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
START_TIME = time.time()

app = Flask(__name__)

# ---- load model + encoders sekali saat startup ----
model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))
le_prov = joblib.load(os.path.join(ARTIFACT_DIR, "le_prov.pkl"))
le_komo = joblib.load(os.path.join(ARTIFACT_DIR, "le_komo.pkl"))

# ---- 11 metrics ----
REQUEST_COUNT = Counter(
    "ml_prediction_requests_total", "Total permintaan prediksi", ["status"]
)
PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds", "Latensi endpoint /predict (detik)"
)
PREDICTED_VALUE = Histogram(
    "ml_predicted_value_ton", "Distribusi nilai hasil prediksi (ton)",
    buckets=(1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, float("inf")),
)
IN_PROGRESS = Gauge(
    "ml_requests_in_progress", "Jumlah request /predict yang sedang diproses"
)
VALIDATION_ERRORS = Counter(
    "ml_validation_errors_total", "Total error validasi input (provinsi/komoditas tidak dikenal)"
)
LAST_PREDICTION_TS = Gauge(
    "ml_last_prediction_timestamp_seconds", "Unix timestamp prediksi terakhir yang berhasil"
)
CPU_USAGE = Gauge("ml_cpu_usage_percent", "Penggunaan CPU proses saat ini (%)")
MEM_USAGE_PCT = Gauge("ml_memory_usage_percent", "Penggunaan memori sistem (%)")
MEM_USAGE_BYTES = Gauge("ml_memory_usage_bytes", "Penggunaan memori proses (bytes)")
DISK_USAGE_PCT = Gauge("ml_disk_usage_percent", "Penggunaan disk pada root filesystem (%)")
UPTIME = Gauge("ml_app_uptime_seconds", "Lama waktu service berjalan (detik)")

_process = psutil.Process(os.getpid())


def _refresh_system_metrics():
    CPU_USAGE.set(_process.cpu_percent(interval=None))
    MEM_USAGE_PCT.set(psutil.virtual_memory().percent)
    MEM_USAGE_BYTES.set(_process.memory_info().rss)
    DISK_USAGE_PCT.set(psutil.disk_usage("/").percent)
    UPTIME.set(time.time() - START_TIME)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", uptime_seconds=round(time.time() - START_TIME, 1))


@app.route("/predict", methods=["POST"])
@PREDICTION_LATENCY.time()
def predict():
    IN_PROGRESS.inc()
    try:
        payload = request.get_json(force=True, silent=True) or {}
        provinsi = payload.get("provinsi")
        komoditas = payload.get("komoditas")

        if not provinsi or not komoditas:
            VALIDATION_ERRORS.inc()
            REQUEST_COUNT.labels(status="error").inc()
            return jsonify(error="Field 'provinsi' dan 'komoditas' wajib diisi"), 400

        if provinsi not in le_prov.classes_:
            VALIDATION_ERRORS.inc()
            REQUEST_COUNT.labels(status="error").inc()
            return jsonify(error=f"Provinsi tidak dikenal: {provinsi}"), 400

        if komoditas not in le_komo.classes_:
            VALIDATION_ERRORS.inc()
            REQUEST_COUNT.labels(status="error").inc()
            return jsonify(error=f"Komoditas tidak dikenal: {komoditas}"), 400

        prov_enc = le_prov.transform([provinsi])[0]
        komo_enc = le_komo.transform([komoditas])[0]
        pred_ton = float(model.predict([[prov_enc, komo_enc]])[0])

        PREDICTED_VALUE.observe(pred_ton)
        LAST_PREDICTION_TS.set(time.time())
        REQUEST_COUNT.labels(status="success").inc()

        return jsonify(
            provinsi=provinsi,
            komoditas=komoditas,
            predicted_produksi_ton=round(pred_ton, 2),
        )
    except Exception as e:
        logger.exception("Prediction failed")
        REQUEST_COUNT.labels(status="error").inc()
        return jsonify(error=str(e)), 500
    finally:
        IN_PROGRESS.dec()


@app.route("/metrics", methods=["GET"])
def metrics():
    _refresh_system_metrics()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
