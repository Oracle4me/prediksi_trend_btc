import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "btc_usd_2021_2026.csv")

DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE = None
BTC_SYMBOL = "BTC-USD"

TREND_HORIZON = 14
FAST_MA = 14
SLOW_MA = 45
REGIME_THRESHOLD = 0.012

# Load data Bitcoin dari CSV lokal atau fetch ulang dari Yahoo Finance.
def load_or_fetch_data(start=DEFAULT_START_DATE, end=DEFAULT_END_DATE, force_update=False):

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE) and not force_update:
        df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()

        if start is not None:
            df = df[df.index >= pd.to_datetime(start)]

        if end is not None:
            df = df[df.index <= pd.to_datetime(end)]

        return df

    try:
        import yfinance as yf

        df = yf.download(
            BTC_SYMBOL,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in required if c in df.columns]].copy()
        df = df.dropna()
        df.index.name = "Date"

        df.to_csv(DATA_FILE)
        return df

    except Exception:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
            return df.set_index("Date").sort_index()

        return None


# Semua fitur hanya memakai data historis.
def add_technical_indicators(df):

    df = df.copy()

    df["Return_1"] = df["Close"].pct_change()
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_7"] = df["Close"].pct_change(7)
    df["Return_14"] = df["Close"].pct_change(14)
    df["Return_30"] = df["Close"].pct_change(30)

    df["Momentum_3"] = df["Close"] - df["Close"].shift(3)
    df["Momentum_7"] = df["Close"] - df["Close"].shift(7)
    df["Momentum_14"] = df["Close"] - df["Close"].shift(14)
    df["Momentum_30"] = df["Close"] - df["Close"].shift(30)

    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_14"] = df["Close"].rolling(14).mean()
    df["MA_30"] = df["Close"].rolling(30).mean()
    df["MA_45"] = df["Close"].rolling(45).mean()
    df["MA_60"] = df["Close"].rolling(60).mean()

    df["MA_7_Slope"] = df["MA_7"].pct_change(7)
    df["MA_14_Slope"] = df["MA_14"].pct_change(14)
    df["MA_30_Slope"] = df["MA_30"].pct_change(14)
    df["MA_45_Slope"] = df["MA_45"].pct_change(14)
    df["MA_60_Slope"] = df["MA_60"].pct_change(14)

    df["Close_MA7_Ratio"] = df["Close"] / df["MA_7"]
    df["Close_MA14_Ratio"] = df["Close"] / df["MA_14"]
    df["Close_MA30_Ratio"] = df["Close"] / df["MA_30"]
    df["Close_MA45_Ratio"] = df["Close"] / df["MA_45"]
    df["Close_MA60_Ratio"] = df["Close"] / df["MA_60"]

    df["MA14_MA45_Spread"] = (df["MA_14"] - df["MA_45"]) / df["MA_45"]
    df["MA30_MA60_Spread"] = (df["MA_30"] - df["MA_60"]) / df["MA_60"]

    df["Volatility_7"] = df["Return_1"].rolling(7).std()
    df["Volatility_14"] = df["Return_1"].rolling(14).std()
    df["Volatility_30"] = df["Return_1"].rolling(30).std()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(14).mean()

    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA_7"] = df["Volume"].rolling(7).mean()
    df["Volume_MA_14"] = df["Volume"].rolling(14).mean()
    df["Volume_MA_30"] = df["Volume"].rolling(30).mean()
    df["Volume_Ratio_7"] = df["Volume"] / df["Volume_MA_7"]
    df["Volume_Ratio_30"] = df["Volume"] / df["Volume_MA_30"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def add_trend_target(
    df,
    horizon=TREND_HORIZON,
    threshold=REGIME_THRESHOLD,
):
    """
    Target binary untuk prediksi trend Bitcoin:

    0 = Turun / tidak naik
    1 = Naik

    Label dibuat berdasarkan return harga pada horizon 14 hari ke depan.
    """

    df = df.copy()

    future_return = (
        df["Close"].shift(-horizon) / df["Close"]
    ) - 1

    df["Future_Return"] = future_return
    df["Trend_Score"] = future_return

    df["Trend_Label"] = np.where(
        future_return > threshold,
        1,
        0,
    ).astype(float)

    # Baris terakhir tidak memiliki data harga masa depan.
    df.loc[future_return.isna(), "Trend_Label"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Future_Return", "Trend_Label"])

    df["Trend_Label"] = df["Trend_Label"].astype(int)

    return df


def get_feature_columns():
    return [
        "Open", "High", "Low", "Close", "Volume",
        "Return_1", "Return_3", "Return_7", "Return_14", "Return_30",
        "Momentum_3", "Momentum_7", "Momentum_14", "Momentum_30",
        "MA_7", "MA_14", "MA_30", "MA_45", "MA_60",
        "MA_7_Slope", "MA_14_Slope", "MA_30_Slope", "MA_45_Slope", "MA_60_Slope",
        "Close_MA7_Ratio", "Close_MA14_Ratio", "Close_MA30_Ratio", "Close_MA45_Ratio", "Close_MA60_Ratio",
        "MA14_MA45_Spread", "MA30_MA60_Spread",
        "Volatility_7", "Volatility_14", "Volatility_30",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "ATR_14",
        "Volume_Change", "Volume_MA_7", "Volume_MA_14", "Volume_MA_30",
        "Volume_Ratio_7", "Volume_Ratio_30"
    ]


def prepare_trend_dataframe(df, trend_horizon=TREND_HORIZON):
    df_ind = add_technical_indicators(df)
    df_trend = add_trend_target(df_ind, horizon=trend_horizon)
    return df_trend


def prepare_ml_data(df, test_size=0.2, trend_horizon=TREND_HORIZON, forecast_horizon=None):
    df_trend = prepare_trend_dataframe(df, trend_horizon=trend_horizon)
    feature_cols = get_feature_columns()

    X = df_trend[feature_cols].values
    y = df_trend["Trend_Label"].astype(int).values
    dates = df_trend.index

    split_idx = int(len(df_trend) * (1 - test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    dates_test = dates[split_idx:]

    return X_train, X_test, y_train, y_test, dates_test, feature_cols


# Data sequence untuk LSTM classifier.
def prepare_lstm_data(df, seq_length=120, test_size=0.2, trend_horizon=TREND_HORIZON, forecast_horizon=None):

    df_trend = prepare_trend_dataframe(df, trend_horizon=trend_horizon)
    feature_cols = get_feature_columns()

    feature_values = df_trend[feature_cols].values
    labels = df_trend["Trend_Label"].astype(int).values
    dates = df_trend.index

    split_idx = int(len(df_trend) * (1 - test_size))

    scaler = MinMaxScaler()
    scaler.fit(feature_values[:split_idx])

    scaled_features = scaler.transform(feature_values)

    X, y, seq_dates = [], [], []

    for i in range(seq_length, len(df_trend)):
        X.append(scaled_features[i - seq_length:i])
        y.append(labels[i])
        seq_dates.append(dates[i])

    X = np.array(X)
    y = np.array(y).astype(int)
    seq_dates = pd.DatetimeIndex(seq_dates)

    train_cutoff_date = dates[split_idx]

    train_mask = seq_dates < train_cutoff_date
    test_mask = seq_dates >= train_cutoff_date

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    dates_test = seq_dates[test_mask]

    return X_train, X_test, y_train, y_test, scaler, dates_test, feature_cols


def latest_lstm_sequence(
    df,
    scaler,
    seq_length=120,
    trend_horizon=TREND_HORIZON,
):
    """
    Membuat sequence terbaru untuk prediksi LSTM.
    Hanya memakai fitur teknikal, tanpa membuat target masa depan.
    """

    df_features = add_technical_indicators(df)
    feature_cols = get_feature_columns()

    if len(df_features) < seq_length:
        raise ValueError(
            f"Data fitur hanya {len(df_features)} baris, "
            f"sedangkan LSTM membutuhkan {seq_length} baris."
        )

    latest_features = (
        df_features[feature_cols]
        .tail(seq_length)
        .values
    )

    latest_scaled = scaler.transform(latest_features)

    return latest_scaled.reshape(
        1,
        seq_length,
        len(feature_cols),
    )