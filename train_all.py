import os
import sys
import json
import joblib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_or_fetch_data, prepare_ml_data, prepare_lstm_data
from models.ml_models import train_random_forest, train_xgboost, train_mlp
from models.lstm_model import train_lstm


# PATH CONFIG
BASE_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")

TEST_SIZE = 0.2
TREND_HORIZON = 14
LSTM_SEQ_LENGTH = 120
EPOCHS = 100
BATCH_SIZE = 16

def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    return value


def format_number(value, decimals=2):
    try:
        value = 0.0 if value is None else float(value)
        if np.isnan(value) or np.isinf(value):
            value = 0.0
        return f"{value:.{decimals}f}"
    except Exception:
        return f"{0:.{decimals}f}"


# Untuk target binary:
# 0 = Turun / tidak naik, 1 = Naik.
def print_class_distribution(y, label="Data"):
    y = np.asarray(y).astype(int)

    if len(y) == 0:
        print(f"      {label:<14}: kosong")
        return

    turun = np.mean(y == 0) * 100
    naik = np.mean(y == 1) * 100

    print(
        f"      {label:<14}: "
        f"Turun/Tidak Naik={turun:.2f}% | "
        f"Naik={naik:.2f}%"
    )


# Print dataset helpers
def print_dataset_info(
    title,
    train_size,
    test_size,
    y_train,
    y_test,
    feature_count=None,
    input_shape=None,
):
    print(f"\n{title}")

    if feature_count is not None:
        print(f"      Jumlah fitur    : {feature_count}")

    if input_shape is not None:
        print(f"      Input shape     : {input_shape}")

    print(f"      Train size      : {train_size}")
    print(f"      Test size       : {test_size}")

    print_class_distribution(y_train, "Train")
    print_class_distribution(y_test, "Test")


# Menampilkan metric klasifikasi utama.
def print_metrics(model_name, metrics):
    print(f"      {model_name} selesai.")
    print(
        "      "
        f"BalancedAcc: {format_number(metrics.get('Balanced_Accuracy', 0), 2)}% | "
        f"Precision: {format_number(metrics.get('Precision', 0), 2)}% | "
        f"Recall: {format_number(metrics.get('Recall', 0), 2)}% | "
        f"F1: {format_number(metrics.get('F1_Score', 0), 2)}%"
    )

    if "Predicted_Bullish_Ratio" in metrics:
        print(
            "      "
            f"Predicted bullish: "
            f"{format_number(metrics.get('Predicted_Bullish_Ratio', 0), 2)}%"
        )

    if "decision_threshold" in metrics:
        print(
            f"      Threshold: "
            f"{format_number(metrics.get('decision_threshold', 0.5), 4)}"
        )

# Menyimpan seluruh hasil metrik ke JSON dan CSV.
def save_summary(metrics_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    safe_metrics_list = [make_json_safe(m) for m in metrics_list]

    with open(os.path.join(save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(safe_metrics_list, f, indent=4)

    pd.DataFrame(safe_metrics_list).to_csv(
        os.path.join(save_dir, "training_summary.csv"),
        index=False,
    )

# Model terbaik ditentukan berdasarkan F1 Score.
# Balanced Accuracy digunakan sebagai pembanding jika F1 Score sama.
def get_best_model(metrics_list):
    sorted_metrics = sorted(
        metrics_list,
        key=lambda metrics: (
            float(metrics.get("F1_Score", 0)),
            float(metrics.get("Balanced_Accuracy", 0)),
        ),
        reverse=True,
    )

    return sorted_metrics[0], sorted_metrics

# Alur:
# load data -> feature engineering -> target trend -> train 4 model -> evaluasi -> simpan hasil -> dashboard.
def main():
    print("=" * 70)
    print("  Bitcoin Trend Prediction - Binary Classification Pipeline")
    print("  Model: LSTM | MLP | XGBoost | Random Forest")
    print("=" * 70)

    print("\nKonfigurasi Eksperimen:")
    print("  Periode Data       : 2021-01-01 - Terbaru")
    print("  Target             : Binary trend, 0=Turun/Tidak Naik, 1=Naik")
    print(f"  Trend Horizon      : {TREND_HORIZON} hari")
    print(f"  Test Size          : {int(TEST_SIZE * 100)}%")
    print(f"  LSTM Sequence      : {LSTM_SEQ_LENGTH} hari")
    print(f"  Epochs             : {EPOCHS}")
    print(f"  Batch Size         : {BATCH_SIZE}")

    print("\n[1/5] Memuat data Bitcoin dari CSV/Yahoo Finance...")

    df = load_or_fetch_data(
        start="2021-01-01",
        end=None,
        force_update=False,
    )

    if df is None or df.empty:
        print("GAGAL: Data tidak dapat dimuat.")
        return

    print(
        f"      Data dimuat: {len(df)} baris | "
        f"{df.index[0].date()} - {df.index[-1].date()}"
    )


    # 2. RANDOM FOREST
    print("\n[2/5] Menyiapkan data fitur klasik dan training Random Forest...")

    X_tr, X_te, y_tr, y_te, dates_te, feats = prepare_ml_data(
        df,
        test_size=TEST_SIZE,
        trend_horizon=TREND_HORIZON,
    )

    print_dataset_info(
        title="      Dataset klasik",
        train_size=len(X_tr),
        test_size=len(X_te),
        y_train=y_tr,
        y_test=y_te,
        feature_count=len(feats),
    )

    _, rf_metrics, _ = train_random_forest(
        X_tr,
        X_te,
        y_tr,
        y_te,
        dates_te,
        tune=False,
        save_dir=SAVE_DIR,
    )

    print_metrics("Random Forest", rf_metrics)

    # 3. XGBOOST
    print("\n[3/5] Training XGBoost...")

    _, xgb_metrics, _ = train_xgboost(
        X_tr,
        X_te,
        y_tr,
        y_te,
        dates_te,
        tune=False,
        save_dir=SAVE_DIR,
    )

    print_metrics("XGBoost", xgb_metrics)

    # 4. MLP
    print("\n[4/5] Training MLP...")

    _, mlp_metrics, _ = train_mlp(
        X_tr,
        X_te,
        y_tr,
        y_te,
        dates_te,
        save_dir=SAVE_DIR,
        epochs=50,
        batch_size=BATCH_SIZE,
    )

    print_metrics("MLP", mlp_metrics)

    # 5. LSTM
    print("\n[5/5] Menyiapkan sequence data dan training LSTM...")

    (
        X_tr_l,
        X_te_l,
        y_tr_l,
        y_te_l,
        scaler,
        dates_te_l,
        lstm_feats,
    ) = prepare_lstm_data(
        df,
        seq_length=LSTM_SEQ_LENGTH,
        test_size=TEST_SIZE,
        trend_horizon=TREND_HORIZON,
    )

    print_dataset_info(
        title="      Dataset LSTM",
        train_size=len(X_tr_l),
        test_size=len(X_te_l),
        y_train=y_tr_l,
        y_test=y_te_l,
        feature_count=len(lstm_feats),
        input_shape=X_tr_l.shape[1:],
    )

    _, lstm_metrics, _, _ = train_lstm(
        X_tr_l,
        X_te_l,
        y_tr_l,
        y_te_l,
        dates_te_l,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR,
        tune=True,
        val_size=0.1,
        threshold_objective="f1",
        use_ensemble=False,
        ensemble_top_k=1,
    )

    print_metrics("LSTM", lstm_metrics)

    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(SAVE_DIR, "lstm_scaler.pkl"))

    all_metrics = [
        make_json_safe(lstm_metrics),
        make_json_safe(mlp_metrics),
        make_json_safe(xgb_metrics),
        make_json_safe(rf_metrics),
    ]

    save_summary(all_metrics, SAVE_DIR)

    best_model, sorted_metrics = get_best_model(all_metrics)

    print("\n" + "=" * 70)
    print("  RINGKASAN HASIL BINARY TREND CLASSIFICATION")
    print("=" * 70)
    print(
        "Model terbaik ditentukan berdasarkan F1 Score, "
        "dengan Balanced Accuracy sebagai pembanding."
    )
    print("-" * 70)

    for m in sorted_metrics:

        star = (
            " ★ TERBAIK"
            if m.get("model") == best_model.get("model")
            else ""
        )

    print(
        f"  {m.get('model', '-'):<15} "
        f"BalancedAcc={format_number(m.get('Balanced_Accuracy', 0), 2):>7}%  "
        f"Precision={format_number(m.get('Precision', 0), 2):>7}%  "
        f"Recall={format_number(m.get('Recall', 0), 2):>7}%  "
        f"F1={format_number(m.get('F1_Score', 0), 2):>7}%"
        f"{star}"
    )

    print("=" * 70)
    print(f"\nModel dan hasil training tersimpan di:\n  {SAVE_DIR}")

    print("\nFile output utama:")
    print("  - rf_model.pkl | rf_predictions.csv | rf_metrics.json")
    print("  - xgb_model.pkl | xgb_predictions.csv | xgb_metrics.json")
    print("  - mlp_model.keras | mlp_scaler_X.pkl | mlp_predictions.csv | mlp_metrics.json | mlp_history.csv")
    print("  - lstm_model.keras | lstm_scaler.pkl | lstm_predictions.csv | lstm_metrics.json")
    print("  - training_summary.json | training_summary.csv")

    print("\nJalankan `python app.py` untuk membuka web dashboard.\n")


if __name__ == "__main__":
    main()