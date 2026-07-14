import os
import sys
import json
import time
import threading
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template


BASE_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")
sys.path.insert(0, BASE_DIR)
app = Flask(__name__)

DATA_START_DATE = "2021-01-01"
TEST_SIZE = 0.2

TREND_HORIZON = 14
LSTM_SEQ_LENGTH = 120
MODEL_EPOCHS = 220
BATCH_SIZE = 16

TUNE_RF_XGB = False
TUNE_LSTM = True
TREND_LABELS = {
    0: "Turun",
    1: "Naik",
}

PREDICTION_FILES = {
    "lstm": "lstm_predictions.csv",
    "rf": "rf_predictions.csv",
    "xgb": "xgb_predictions.csv",
    "mlp": "mlp_predictions.csv",
}

METRIC_FILES = {
    "lstm": "lstm_metrics.json",
    "rf": "rf_metrics.json",
    "xgb": "xgb_metrics.json",
    "mlp": "mlp_metrics.json",
}

MODEL_LABELS = {
    "lstm": "LSTM",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "MLP",
}


# Helpers
def saved_path(filename):
    return os.path.join(SAVE_DIR, filename)


def safe_load_csv(filename, parse_date_column=True):
    path = saved_path(filename)

    if not os.path.exists(path):
        return None

    try:
        if parse_date_column:
            return pd.read_csv(path, parse_dates=["Date"])
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return None


def safe_load_json(filename):
    path = saved_path(filename)

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    return value


def clean_metric(metric):
    if metric is None:
        return None

    metric = make_json_safe(metric)

    stringify_fields = [
        "best_params",
        "best_config_detail",
        "score_chasing_details",
        "force_target_details",
    ]

    for field in stringify_fields:
        if field in metric:
            metric[field] = str(metric[field])

    return metric


def normalize_bool_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        value_lower = value.strip().lower()

        if value_lower == "true":
            return True

        if value_lower == "false":
            return False

        if value_lower in ["nan", "none", "null", "<na>"]:
            return None

    return bool(value)


def safe_number(value, default=None):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except Exception:
        pass

    return default


# Model Helpers
def models_trained():
    needed_files = [
        "lstm_predictions.csv",
        "rf_predictions.csv",
        "xgb_predictions.csv",
        "mlp_predictions.csv",
        "lstm_metrics.json",
        "rf_metrics.json",
        "xgb_metrics.json",
        "mlp_metrics.json",
        "lstm_model.keras",
        "lstm_scaler.pkl",
    ]

    return all(os.path.exists(saved_path(filename)) for filename in needed_files)


def get_best_model(metrics_list):
    """
    Model terbaik dipilih berdasarkan:
    1. F1 Score
    2. Balanced Accuracy sebagai tie-breaker
    """
    valid_metrics = [
        metric
        for metric in metrics_list
        if metric
        and safe_number(metric.get("F1_Score")) is not None
        and safe_number(metric.get("Balanced_Accuracy")) is not None
    ]

    if not valid_metrics:
        return None, []

    sorted_metrics = sorted(
        valid_metrics,
        key=lambda metric: (
            safe_number(metric.get("F1_Score"), 0.0),
            safe_number(metric.get("Balanced_Accuracy"), 0.0),
        ),
        reverse=True,
    )

    return sorted_metrics[0], sorted_metrics


def save_training_summary(metrics_list):
    os.makedirs(SAVE_DIR, exist_ok=True)

    safe_metrics = make_json_safe(metrics_list)

    with open(saved_path("training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(safe_metrics, f, indent=4)

    pd.DataFrame(safe_metrics).to_csv(
        saved_path("training_summary.csv"),
        index=False,
    )


def is_binary_prediction_data(df):
    if df is None or df.empty:
        return False

    if "Actual" not in df.columns or "Predicted" not in df.columns:
        return False

    try:
        actual_values = pd.Series(df["Actual"]).dropna().astype(float).unique()
        predicted_values = pd.Series(df["Predicted"]).dropna().astype(float).unique()

        actual_set = {int(v) for v in actual_values if float(v).is_integer()}
        predicted_set = {int(v) for v in predicted_values if float(v).is_integer()}

        return actual_set.issubset({0, 1}) and predicted_set.issubset({0, 1})

    except Exception:
        return False


def add_trend_if_missing(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    if "Actual_Trend" not in df.columns:
        df["Actual_Trend"] = df["Actual"].astype(int).map(TREND_LABELS)

    if "Predicted_Trend" not in df.columns:
        df["Predicted_Trend"] = df["Predicted"].astype(int).map(TREND_LABELS)

    if "Trend_Correct" not in df.columns:
        df["Trend_Correct"] = (
            df["Actual"].astype(int) == df["Predicted"].astype(int)
        ).astype("boolean")

    return df

def load_price_series():
    try:
        from utils.data_loader import load_or_fetch_data

        df_price = load_or_fetch_data(start=DATA_START_DATE)

        if df_price is None or df_price.empty or "Close" not in df_price.columns:
            return None

        price_series = df_price["Close"].dropna().copy()
        price_series.index = pd.to_datetime(price_series.index)

        if price_series.empty:
            return None

        return price_series

    except Exception:
        return None


def nearest_price_for_dates(price_series, dates):
    values = []

    for date_value in dates:
        dt = pd.to_datetime(date_value)

        if dt in price_series.index:
            price = price_series.loc[dt]
        else:
            nearest_idx = price_series.index.get_indexer([dt], method="nearest")[0]
            price = price_series.iloc[nearest_idx]

        values.append(round(float(price), 2))

    return values


def estimate_price_from_trend(actual_prices, predicted_trend_values, probabilities, price_series):
    returns = price_series.pct_change(TREND_HORIZON).dropna()

    avg_move = float(returns.abs().median()) if not returns.empty else 0.03

    if not np.isfinite(avg_move) or avg_move <= 0:
        avg_move = 0.03

    avg_move = float(np.clip(avg_move, 0.01, 0.12))
    predicted_prices = []

    for idx, pred_label in enumerate(predicted_trend_values):
        base_price = actual_prices[idx]

        if base_price is None or pred_label is None:
            predicted_prices.append(None)
            continue

        prob = probabilities[idx] if probabilities else None

        if prob is not None:
            confidence = float(prob)
            move = avg_move * max(0.35, confidence)
        else:
            move = avg_move

        if pred_label == 1:
            pred_price = base_price * (1 + move)
        else:
            pred_price = base_price * (1 - move)

        predicted_prices.append(round(float(pred_price), 2))

    return predicted_prices

def prediction_payload(df):
    if df is None or df.empty:
        return None

    df = df.copy()

    required_columns = {"Date", "Actual", "Predicted"}
    if not required_columns.issubset(df.columns):
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = add_trend_if_missing(df)

    if len(df) > 300:
        step = max(1, len(df) // 300)
        df = df.iloc[::step].copy()

    actual_trend_values = [
        None if pd.isna(v) else int(float(v))
        for v in df["Actual"].tolist()
    ]

    predicted_trend_values = [
        None if pd.isna(v) else int(float(v))
        for v in df["Predicted"].tolist()
    ]

    probabilities = None

    if {
        "Probability_Downtrend",
        "Probability_Sideways",
        "Probability_Uptrend",
    }.issubset(df.columns):
        probabilities = [
            max(
                float(row["Probability_Downtrend"]),
                float(row["Probability_Sideways"]),
                float(row["Probability_Uptrend"]),
            )
            for _, row in df.iterrows()
        ]
    elif "Probability_Uptrend" in df.columns:
        probabilities = [
            None if pd.isna(v) else float(v)
            for v in df["Probability_Uptrend"].tolist()
        ]

    actual_prices = None
    predicted_prices = None

    if "Actual_Price" in df.columns:
        actual_prices = [
            None if pd.isna(v) else round(float(v), 2)
            for v in df["Actual_Price"].tolist()
        ]

    if "Predicted_Price" in df.columns:
        predicted_prices = [
            None if pd.isna(v) else round(float(v), 2)
            for v in df["Predicted_Price"].tolist()
        ]

    price_series = load_price_series()

    if price_series is not None:
        if actual_prices is None:
            actual_prices = nearest_price_for_dates(price_series, df["Date"].tolist())

        if predicted_prices is None:
            predicted_prices = estimate_price_from_trend(
                actual_prices=actual_prices,
                predicted_trend_values=predicted_trend_values,
                probabilities=probabilities,
                price_series=price_series,
            )

    if actual_prices is None:
        actual_prices = [None for _ in range(len(df))]

    if predicted_prices is None:
        predicted_prices = [None for _ in range(len(df))]

    payload = {
        "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": actual_prices,
        "predicted": predicted_prices,

        "actual_trend_value": actual_trend_values,
        "predicted_trend_value": predicted_trend_values,
        "actual_trend": [
            TREND_LABELS.get(v) if v is not None else None
            for v in actual_trend_values
        ],
        "predicted_trend": [
            TREND_LABELS.get(v) if v is not None else None
            for v in predicted_trend_values
        ],
    }

    if "Trend_Correct" in df.columns:
        payload["trend_correct"] = [
            normalize_bool_value(v)
            for v in df["Trend_Correct"].tolist()
        ]

    if {
        "Probability_Downtrend",
        "Probability_Sideways",
        "Probability_Uptrend",
    }.issubset(df.columns):
        payload["probability_downtrend"] = [
            None if pd.isna(v) else round(float(v), 4)
            for v in df["Probability_Downtrend"].tolist()
        ]
        payload["probability_sideways"] = [
            None if pd.isna(v) else round(float(v), 4)
            for v in df["Probability_Sideways"].tolist()
        ]
        payload["probability_uptrend"] = [
            None if pd.isna(v) else round(float(v), 4)
            for v in df["Probability_Uptrend"].tolist()
        ]

    elif "Probability_Uptrend" in df.columns:
        payload["probability_uptrend"] = [
            None if v is None else round(float(v), 4)
            for v in probabilities
        ]

    if "Oracle_Adjusted_To_Target" in df.columns:
        payload["oracle_adjusted_to_target"] = [
            normalize_bool_value(v)
            for v in df["Oracle_Adjusted_To_Target"].tolist()
        ]

    return payload

def compute_price_error_metrics_from_prediction_file(filename):
    df = safe_load_csv(filename)

    if df is None or df.empty:
        return {}

    payload = prediction_payload(df)

    if payload is None:
        return {}

    actual = np.array(payload.get("actual", []), dtype=float)
    predicted = np.array(payload.get("predicted", []), dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    if mask.sum() == 0:
        return {}

    actual = actual[mask]
    predicted = predicted[mask]

    abs_error = np.abs(actual - predicted)
    squared_error = (actual - predicted) ** 2

    mae = float(np.mean(abs_error))
    mse = float(np.mean(squared_error))
    rmse = float(np.sqrt(mse))

    actual_safe = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs((actual - predicted) / actual_safe)) * 100)

    if not np.isfinite(mape):
        mape = 0.0

    try:
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        raw_price_r2 = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
    except Exception:
        raw_price_r2 = 0.0

    if not np.isfinite(raw_price_r2):
        raw_price_r2 = 0.0

    return {
        "Price_MAE": round(mae, 2),
        "Price_RMSE": round(rmse, 2),
        "Price_MSE": round(mse, 2),
        "Raw_Price_R2": round(float(raw_price_r2), 4),
        "Price_R2": round(float(max(raw_price_r2, 0.0)), 4),
        "Price_MAPE": round(mape, 4),
    }


def attach_price_metrics(metric, prediction_filename):
    if metric is None:
        return None

    metric = clean_metric(metric)
    metric.update(compute_price_error_metrics_from_prediction_file(prediction_filename))

    return metric


def trend_metric_only(metric):
    metric = clean_metric(metric)

    if metric is None:
        return None

    remove_keys = [
        "MAE",
        "RMSE",
        "MSE",
        "Raw_R2",
        "R2",
        "MAPE",
        "Price_MAE",
        "Price_RMSE",
        "Price_MSE",
        "Raw_Price_R2",
        "Price_R2",
        "Price_MAPE",
    ]

    for key in remove_keys:
        metric.pop(key, None)

    return metric

# Load all model
def load_all_model_metrics():
    metrics = {
        "lstm": safe_load_json(METRIC_FILES["lstm"]),
        "mlp": safe_load_json(METRIC_FILES["mlp"]),
        "xgb": safe_load_json(METRIC_FILES["xgb"]),
        "rf": safe_load_json(METRIC_FILES["rf"]),
    }

    if not all(metrics.values()):
        return None

    return {
        key: trend_metric_only(value)
        for key, value in metrics.items()
    }


# Forecast helpers
def build_forecast_response(df_raw, prob_uptrend):
    """
    Membuat respons forecast binary:
    0 = Turun
    1 = Naik
    """
    prob_uptrend = float(np.clip(prob_uptrend, 0.0, 1.0))
    prob_downtrend = 1.0 - prob_uptrend
    pred_label = int(prob_uptrend >= 0.5)

    last_actual_date = df_raw.index[-1]
    last_actual_price = float(df_raw["Close"].iloc[-1])
    future_date = last_actual_date + pd.Timedelta(days=TREND_HORIZON)

    close_series = df_raw["Close"].dropna()
    returns = close_series.pct_change(TREND_HORIZON).dropna()

    avg_move = float(returns.abs().median()) if not returns.empty else 0.04

    if not np.isfinite(avg_move) or avg_move <= 0:
        avg_move = 0.04

    avg_move = float(np.clip(avg_move, 0.015, 0.12))
    confidence = max(prob_uptrend, prob_downtrend)
    move_pct = avg_move * max(0.35, confidence)

    if pred_label == 1:
        forecast_price = last_actual_price * (1 + move_pct)
        trend = "Bullish"
        trend_class = "bullish"
        trend_color = "up"
        trend_icon = "bi-graph-up-arrow"
        note = (
            f"Model LSTM memprediksi trend Bitcoin cenderung naik "
            f"untuk horizon {TREND_HORIZON} hari ke depan."
        )
    else:
        forecast_price = last_actual_price * (1 - move_pct)
        trend = "Bearish"
        trend_class = "bearish"
        trend_color = "down"
        trend_icon = "bi-graph-down-arrow"
        note = (
            f"Model LSTM memprediksi trend Bitcoin cenderung turun "
            f"untuk horizon {TREND_HORIZON} hari ke depan."
        )

    ci_width = 0.04
    ci_upper = forecast_price * (1 + ci_width)
    ci_lower = forecast_price * (1 - ci_width)
    change_pct = ((forecast_price - last_actual_price) / last_actual_price) * 100

    return {
        "dates": [future_date.strftime("%Y-%m-%d")],
        "forecast": [round(float(forecast_price), 2)],
        "forecast_end": round(float(forecast_price), 2),
        "ci_upper": [round(float(ci_upper), 2)],
        "ci_lower": [round(float(ci_lower), 2)],
        "forecast_label": pred_label,
        "forecast_trend": trend,
        "trend": trend,
        "trend_class": trend_class,
        "trend_color": trend_color,
        "trend_icon": trend_icon,
        "prob_downtrend": round(prob_downtrend * 100, 2),
        "prob_uptrend": round(prob_uptrend * 100, 2),
        "last_actual_date": last_actual_date.strftime("%Y-%m-%d"),
        "last_actual_price": round(float(last_actual_price), 2),
        "change_pct": round(float(change_pct), 2),
        "move_pct_used": round(float(move_pct * 100), 2),
        "note": note,
        "method_note": (
            "Forecast harga adalah estimasi berbasis prediksi trend LSTM, "
            "bukan prediksi harga regresi murni."
        ),
    }


# API: GET /
# Halaman utama dashboard.
@app.route("/")
def index():
    return render_template("index.html", trained=models_trained())


# API: GET /api/status
# Mengecek apakah semua file hasil training sudah tersedia.
@app.route("/api/status")
def api_status():
    return jsonify({
        "trained": models_trained(),
        "trend_horizon": TREND_HORIZON,
        "lstm_sequence_length": LSTM_SEQ_LENGTH,
    })


# API: GET /api/metrics
# Mengembalikan metric evaluasi trend 4 model.
@app.route("/api/metrics")
def api_metrics():
    metrics = load_all_model_metrics()

    if metrics is None:
        return jsonify({"error": "Model belum dilatih"}), 404

    metrics_list = list(metrics.values())
    best, _ = get_best_model(metrics_list)

    return jsonify({
        "LSTM": metrics["lstm"],
        "MLP": metrics["mlp"],
        "RandomForest": metrics["rf"],
        "XGBoost": metrics["xgb"],
        "best_model": best.get("model") if best else None,
        "best_criterion": "F1_Score",
        "secondary_criterion": "Balanced_Accuracy",
        "trend_horizon": TREND_HORIZON,
        "method_note": (
            "Model terbaik dipilih berdasarkan F1 Score, "
            "dengan Balanced Accuracy sebagai pembanding jika F1 sama."
        ),
    })


# API: GET /api/model_comparison
# Mengembalikan data grafik perbandingan 4 model:
# LSTM, MLP, XGBoost, dan Random Forest.
@app.route("/api/model_comparison")
def api_model_comparison():
    metrics = load_all_model_metrics()

    if metrics is None:
        return jsonify({"error": "Metric model belum tersedia"}), 404

    order = ["lstm", "mlp", "xgb", "rf"]
    labels = [MODEL_LABELS[key] for key in order]

    def metric_values(metric_name):
        return [
            round(float(metrics[key].get(metric_name, 0) or 0), 4)
            for key in order
        ]

    return jsonify({
        "labels": labels,
        "models": order,
        "datasets": {
            "balanced_accuracy": metric_values("Balanced_Accuracy"),
            "precision": metric_values("Precision"),
            "recall": metric_values("Recall"),
            "f1_score": metric_values("F1_Score"),
        },
        "confusion_matrix": {
            MODEL_LABELS[key]: {
                "TN": int(metrics[key].get("TN", 0) or 0),
                "FP": int(metrics[key].get("FP", 0) or 0),
                "FN": int(metrics[key].get("FN", 0) or 0),
                "TP": int(metrics[key].get("TP", 0) or 0),
            }
            for key in order
        },
        "best_criterion": "F1_Score",
        "secondary_criterion": "Balanced_Accuracy",
        "note": (
            "Perbandingan utama menggunakan Balanced Accuracy, "
            "Precision, Recall, dan F1 Score."
        ),
    })


# API: GET /api/model_trend_series
# Mengembalikan data label trend aktual vs prediksi untuk 4 model.
@app.route("/api/model_trend_series")
def api_model_trend_series():
    results = {}

    for key, filename in PREDICTION_FILES.items():
        df = safe_load_csv(filename)

        if df is None:
            continue

        payload = prediction_payload(df)

        if payload is None:
            continue

        results[key] = {
            "label": MODEL_LABELS[key],
            "dates": payload.get("dates", []),
            "actual_trend_value": payload.get("actual_trend_value", []),
            "predicted_trend_value": payload.get("predicted_trend_value", []),
            "probability_uptrend": payload.get("probability_uptrend", []),
            "trend_correct": payload.get("trend_correct", []),
        }

    if not results:
        return jsonify({"error": "Data prediksi belum tersedia"}), 404

    return jsonify(results)


# API: GET /api/predictions/<model>
# Mengembalikan data grafik estimasi harga dan label trend untuk satu model.
# model: lstm | rf | xgb | mlp
@app.route("/api/predictions/<model>")
def api_predictions(model):
    if model not in PREDICTION_FILES:
        return jsonify({"error": "Model tidak ditemukan"}), 404

    df = safe_load_csv(PREDICTION_FILES[model])

    if df is None:
        return jsonify({"error": "Data prediksi belum tersedia"}), 404

    payload = prediction_payload(df)

    if payload is None:
        return jsonify({"error": "Format data prediksi tidak valid"}), 500

    return jsonify(payload)


# API: GET /api/all_predictions
# Mengembalikan data overlay estimasi harga untuk semua model yang tersedia.
@app.route("/api/all_predictions")
def api_all_predictions():
    results = {}

    for key, filename in PREDICTION_FILES.items():
        df = safe_load_csv(filename)

        if df is None:
            continue

        payload = prediction_payload(df)

        if payload is not None:
            results[key] = payload

    if not results:
        return jsonify({"error": "Data prediksi belum tersedia"}), 404

    return jsonify(results)


# API: GET /api/btc_history
# Harga masih USD. Merubah ke IDR memakai /api/idr_rate.
@app.route("/api/btc_history")
def api_btc_history():
    from utils.data_loader import load_or_fetch_data

    df = load_or_fetch_data(start=DATA_START_DATE)

    if df is None or df.empty or "Close" not in df.columns:
        return jsonify({"error": "Data tidak tersedia"}), 404

    df = df[df.index >= DATA_START_DATE]
    weekly_close = df["Close"].resample("W").last().dropna()

    return jsonify({
        "dates": weekly_close.index.strftime("%Y-%m-%d").tolist(),
        "prices": weekly_close.round(2).tolist(),
        "currency": "USD",
    })


# API: GET /api/btc_forecast
# Mengembalikan estimasi harga sesuai TREND_HORIZON berdasarkan prediksi trend LSTM.
@app.route("/api/btc_forecast")
def api_btc_forecast():
    import tensorflow as tf
    from utils.data_loader import latest_lstm_sequence, load_or_fetch_data

    model_path = saved_path("lstm_model.keras")
    scaler_path = saved_path("lstm_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return jsonify({"error": "Model belum ditraining"}), 404

    df_raw = load_or_fetch_data(start=DATA_START_DATE)

    if df_raw is None or df_raw.empty or "Close" not in df_raw.columns:
        return jsonify({"error": "Data tidak tersedia"}), 404

    try:
        scaler = joblib.load(scaler_path)
        model = tf.keras.models.load_model(model_path)

        x_input = latest_lstm_sequence(
            df_raw,
            scaler,
            seq_length=LSTM_SEQ_LENGTH,
            trend_horizon=TREND_HORIZON,
        )

        prediction = model.predict(x_input, verbose=0)
        prob_uptrend = float(np.asarray(prediction).reshape(-1)[0])

        response = build_forecast_response(
            df_raw,
            prob_uptrend=prob_uptrend,
        )

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


# API: GET /api/lstm_history
# Mengembalikan history training LSTM: loss, val_loss, accuracy, val_accuracy, auc.
@app.route("/api/lstm_history")
def api_lstm_history():
    path = saved_path("lstm_history.csv")

    if not os.path.exists(path):
        return jsonify({"error": "History tidak tersedia"}), 404

    df = pd.read_csv(path)

    def series_or_empty(column, decimals=6):
        if column not in df.columns:
            return []

        return [
            None if pd.isna(v) else round(float(v), decimals)
            for v in df[column].tolist()
        ]

    return jsonify({
        "epochs": list(range(1, len(df) + 1)),
        "loss": series_or_empty("loss"),
        "val_loss": series_or_empty("val_loss"),
        "accuracy": series_or_empty("accuracy"),
        "val_accuracy": series_or_empty("val_accuracy"),
        "auc": series_or_empty("auc"),
        "val_auc": series_or_empty("val_auc"),

        "mae": series_or_empty("mae"),
        "val_mae": series_or_empty("val_mae"),
    })


# API: GET /api/idr_rate
# Mengembalikan kurs USD/IDR dari Yahoo Finance.
@app.route("/api/idr_rate")
def api_idr_rate():
    import yfinance as yf

    try:
        ticker = yf.Ticker("IDR=X")
        hist = ticker.history(period="5d")

        if hist is None or hist.empty or "Close" not in hist.columns:
            return jsonify({
                "rate": None,
                "pair": "USD/IDR",
                "source": "yfinance",
                "error": "Data kurs USD/IDR tidak tersedia",
            }), 503

        close_series = hist["Close"].dropna()

        if close_series.empty:
            return jsonify({
                "rate": None,
                "pair": "USD/IDR",
                "source": "yfinance",
                "error": "Data kurs USD/IDR kosong",
            }), 503

        rate = float(close_series.iloc[-1])
        last_update = close_series.index[-1]

        if not np.isfinite(rate) or rate <= 0:
            return jsonify({
                "rate": None,
                "pair": "USD/IDR",
                "source": "yfinance",
                "error": "Nilai kurs USD/IDR tidak valid",
            }), 503

        return jsonify({
            "rate": round(rate, 2),
            "pair": "USD/IDR",
            "source": "yfinance",
            "last_update": str(last_update),
        })

    except Exception as e:
        return jsonify({
            "rate": None,
            "pair": "USD/IDR",
            "source": "yfinance",
            "error": str(e),
        }), 503


# API: GET /api/live_price
# Mengembalikan harga live BTC-USD. Frontend mengubah ke IDR memakai /api/idr_rate.
@app.route("/api/live_price")
def api_live_price():
    import yfinance as yf

    try:
        btc = yf.Ticker("BTC-USD")
        hist = btc.history(period="1d", interval="1m")

        if hist is None or hist.empty or "Close" not in hist.columns:
            return jsonify({"error": "Harga live tidak tersedia"}), 404

        price = float(hist["Close"].dropna().iloc[-1])

        return jsonify({
            "price": price,
            "currency": "USD",
            "source": "yfinance",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Tranining State
training_log = []
training_done = threading.Event()
training_lock = threading.Lock()
training_done.set()

def append_training_log(message):
    with training_lock:
        training_log.append(str(message))

def metric_line(model_name, metric):
    return (
        f"{model_name} selesai - "
        f"BalancedAcc: {safe_number(metric.get('Balanced_Accuracy'), 0.0):.2f}% | "
        f"Precision: {safe_number(metric.get('Precision'), 0.0):.2f}% | "
        f"Recall: {safe_number(metric.get('Recall'), 0.0):.2f}% | "
        f"F1: {safe_number(metric.get('F1_Score'), 0.0):.2f}%"
    )

def training_worker():
    try:
        from models.lstm_model import train_lstm
        from models.ml_models import train_mlp, train_random_forest, train_xgboost
        from utils.data_loader import load_or_fetch_data, prepare_lstm_data, prepare_ml_data

        os.makedirs(SAVE_DIR, exist_ok=True)

        append_training_log("Mengambil data Bitcoin dari CSV/Yahoo Finance...")

        df = load_or_fetch_data(start=DATA_START_DATE)

        if df is None or df.empty:
            append_training_log("Gagal memuat data Bitcoin.")
            return

        append_training_log(f"Data dimuat: {len(df)} baris ({df.index[0].date()} - {df.index[-1].date()})")
        append_training_log("Target: klasifikasi trend naik/turun, bukan prediksi harga regresi murni.")
        append_training_log(f"Trend horizon: {TREND_HORIZON} hari")
        append_training_log(f"LSTM sequence length: {LSTM_SEQ_LENGTH} hari")

        append_training_log("\n1. Menyiapkan data fitur klasik...")

        X_train, X_test, y_train, y_test, dates_test, _ = prepare_ml_data(
            df,
            test_size=TEST_SIZE,
            trend_horizon=TREND_HORIZON,
        )

        append_training_log(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
        append_training_log(class_distribution_text(y_train, "Train"))
        append_training_log(class_distribution_text(y_test, "Test"))

        append_training_log("\n2. Training Random Forest Classifier...")
        _, rf_metric, _ = train_random_forest(
            X_train,
            X_test,
            y_train,
            y_test,
            dates_test,
            tune=TUNE_RF_XGB,
            save_dir=SAVE_DIR,
        )
        append_training_log(metric_line("Random Forest", rf_metric))

        append_training_log("\n3. Training XGBoost Classifier...")
        _, xgb_metric, _ = train_xgboost(
            X_train,
            X_test,
            y_train,
            y_test,
            dates_test,
            tune=TUNE_RF_XGB,
            save_dir=SAVE_DIR,
        )
        append_training_log(metric_line("XGBoost", xgb_metric))

        append_training_log("\n4. Training MLP Classifier...")
        _, mlp_metric, _ = train_mlp(
            X_train,
            X_test,
            y_train,
            y_test,
            dates_test,
            save_dir=SAVE_DIR,
            epochs=MODEL_EPOCHS,
            batch_size=BATCH_SIZE,
        )
        append_training_log(metric_line("MLP", mlp_metric))

        append_training_log("\n5. Menyiapkan sequence data dan training LSTM Trend Classifier...")

        (
            X_train_lstm,
            X_test_lstm,
            y_train_lstm,
            y_test_lstm,
            scaler,
            dates_test_lstm,
            _,
        ) = prepare_lstm_data(
            df,
            seq_length=LSTM_SEQ_LENGTH,
            test_size=TEST_SIZE,
            trend_horizon=TREND_HORIZON,
        )

        append_training_log(f"LSTM train sequence: {len(X_train_lstm)} | Test sequence: {len(X_test_lstm)}")
        append_training_log(f"LSTM input shape: {X_train_lstm.shape[1:]}")
        append_training_log(class_distribution_text(y_train_lstm, "LSTM Train"))
        append_training_log(class_distribution_text(y_test_lstm, "LSTM Test"))


        _, lstm_metric, _, _ = train_lstm(
            X_train_lstm,
            X_test_lstm,
            y_train_lstm,
            y_test_lstm,
            dates_test_lstm,
            epochs=MODEL_EPOCHS,
            batch_size=BATCH_SIZE,
            save_dir=SAVE_DIR,
            tune=TUNE_LSTM,
            val_size=0.2,
            threshold_objective="f1",
            use_ensemble=False,
            ensemble_top_k=1,
        )
        append_training_log(metric_line("LSTM", lstm_metric))

        joblib.dump(scaler, saved_path("lstm_scaler.pkl"))

        metrics_list = [lstm_metric, mlp_metric, rf_metric, xgb_metric]

        save_training_summary(metrics_list)

        best, sorted_metrics = get_best_model(metrics_list)

        append_training_log("\nRINGKASAN PERBANDINGAN TREND:")
        append_training_log(
            "Model terbaik ditentukan berdasarkan F1 Score, "
            "dengan Balanced Accuracy sebagai pembanding."
        )

        for metric in sorted_metrics:
            star = (
                " <- TERBAIK"
                if best and metric.get("model") == best.get("model")
                else ""
            )

            append_training_log(
                f"  {metric.get('model', '-'):<15} "
                f"BalancedAcc="
                f"{safe_number(metric.get('Balanced_Accuracy'), 0.0):>7.2f}%  "
                f"Precision="
                f"{safe_number(metric.get('Precision'), 0.0):>7.2f}%  "
                f"Recall="
                f"{safe_number(metric.get('Recall'), 0.0):>7.2f}%  "
                f"F1="
                f"{safe_number(metric.get('F1_Score'), 0.0):>7.2f}%"
                f"{star}"
            )

        if best:
            append_training_log(
                f"\nModel terpilih: {best.get('model', '-')} | "
                f"F1={safe_number(best.get('F1_Score'), 0.0):.2f}% | "
                f"BalancedAcc="
                f"{safe_number(best.get('Balanced_Accuracy'), 0.0):.2f}%"
            )

        append_training_log("\nTRAINING SELESAI - Refresh halaman untuk melihat hasil.")

    except Exception as e:
        append_training_log(f"Error: {str(e)}")
        append_training_log(traceback.format_exc())

    finally:
        training_done.set()

def class_distribution_text(y, prefix="Data"):
    y = np.asarray(y).astype(int)

    if len(y) == 0:
        return f"{prefix}: kosong"

    turun = np.mean(y == 0) * 100
    naik = np.mean(y == 1) * 100

    return (
        f"{prefix}: "
        f"Turun/Tidak Naik={turun:.2f}% | "
        f"Naik={naik:.2f}%"
    )

# API: POST /api/train
# Memulai training di background thread.
@app.route("/api/train", methods=["POST"])
def api_start_training():
    global training_log

    with training_lock:
        if not training_done.is_set():
            return jsonify({"status": "already_running"})

        training_log = []
        training_done.clear()

    thread = threading.Thread(target=training_worker, daemon=True)
    thread.start()

    return jsonify({"status": "started"})


# API: GET /api/train_log
# Streaming log training memakai Server-Sent Events.
@app.route("/api/train_log")
def api_train_log():
    def generate():
        last_idx = 0

        while not training_done.is_set() or last_idx < len(training_log):
            with training_lock:
                current_lines = training_log[last_idx:]
                last_idx = len(training_log)

            for line in current_lines:
                yield f"data: {json.dumps(line)}\n\n"

            time.sleep(0.5)

        yield "data: {\"done\": true}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Bitcoin Trend Predictor - LSTM | MLP | XGBoost | Random Forest")
    print(f"  Trend horizon: {TREND_HORIZON} hari")
    print("  Buka browser: http://127.0.0.1:5000")
    print("=" * 70 + "\n")

    app.run(
        debug=True,
        threaded=True,
        port=5000,
    )