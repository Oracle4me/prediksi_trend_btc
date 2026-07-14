import os
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


TREND_LABELS = {
    0: "Turun",
    1: "Naik",
}


# METRICS
# Menghitung evaluasi binary classification.
def compute_classification_metrics(
    y_actual,
    y_pred,
    name,
    y_prob=None,
):
    y_actual = np.asarray(y_actual).astype(int).flatten()
    y_pred = np.asarray(y_pred).astype(int).flatten()

    cm = confusion_matrix(
        y_actual,
        y_pred,
        labels=[0, 1],
    )

    metrics = {
        "model": name,

        "Trend_Accuracy": round(
            float(
                accuracy_score(
                    y_actual,
                    y_pred,
                ) * 100
            ),
            4,
        ),

        "Balanced_Accuracy": round(
            float(
                balanced_accuracy_score(
                    y_actual,
                    y_pred,
                ) * 100
            ),
            4,
        ),

        "Precision": round(
            float(
                precision_score(
                    y_actual,
                    y_pred,
                    zero_division=0,
                ) * 100
            ),
            4,
        ),

        "Recall": round(
            float(
                recall_score(
                    y_actual,
                    y_pred,
                    zero_division=0,
                ) * 100
            ),
            4,
        ),

        "F1_Score": round(
            float(
                f1_score(
                    y_actual,
                    y_pred,
                    zero_division=0,
                ) * 100
            ),
            4,
        ),

        "Actual_Bullish_Ratio": round(
            float(
                np.mean(y_actual == 1) * 100
            ),
            4,
        ),

        "Predicted_Bullish_Ratio": round(
            float(
                np.mean(y_pred == 1) * 100
            ),
            4,
        ),

        "TN": int(cm[0][0]),
        "FP": int(cm[0][1]),
        "FN": int(cm[1][0]),
        "TP": int(cm[1][1]),

        "Confusion_Matrix": cm.tolist(),
        "Confusion_Labels": ["Turun", "Naik"],
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)

        # Binary model bisa mengirim:
        # 1D = probabilitas kelas naik
        # 2D = predict_proba dengan kolom [turun, naik]
        if y_prob.ndim == 2:
            if y_prob.shape[1] < 2:
                raise ValueError(
                    "Probability binary harus memiliki minimal 2 kolom."
                )

            probability_up = y_prob[:, 1]

        else:
            probability_up = y_prob.flatten()

        metrics["Avg_Probability_Up"] = round(
            float(
                np.mean(probability_up)
            ),
            4,
        )

    return metrics

def add_trend_columns(pred_df):
    pred_df = pred_df.copy()

    pred_df["Actual_Trend"] = (
        pred_df["Actual"]
        .astype(int)
        .map(TREND_LABELS)
    )

    pred_df["Predicted_Trend"] = (
        pred_df["Predicted"]
        .astype(int)
        .map(TREND_LABELS)
    )

    pred_df["Trend_Correct"] = (
        pred_df["Actual"].astype(int)
        == pred_df["Predicted"].astype(int)
    ).astype("boolean")

    return pred_df


# SAVE OUTPUT
# Menyimpan model, prediksi, dan metric.
def save_prediction_and_metrics(
    model,
    metrics,
    pred_df,
    save_dir,
    model_filename,
    prediction_filename,
    metrics_filename,
):
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    joblib.dump(
        model,
        os.path.join(
            save_dir,
            model_filename,
        ),
    )

    pred_df.to_csv(
        os.path.join(
            save_dir,
            prediction_filename,
        ),
        index=False,
    )

    with open(
        os.path.join(
            save_dir,
            metrics_filename,
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )


# Train Random Forest
def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    dates_test,
    tune=False,
    save_dir="models",
):
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    y_train = np.asarray(
        y_train
    ).astype(int).flatten()

    y_test = np.asarray(
        y_test
    ).astype(int).flatten()

    if tune:
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [8, 12, 18, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", None],
        }

        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        )

        tscv = TimeSeriesSplit(
            n_splits=3
        )

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=tscv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(
            X_train,
            y_train,
        )

        model = (
            grid_search.best_estimator_
        )

        best_params = (
            grid_search.best_params_
        )

    else:
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        best_params = {}

    probability_matrix = (
        model.predict_proba(
            X_test
        )
    )

    probability_up = (
        probability_matrix[:, 1]
    )

    y_pred = (
        probability_up >= 0.5
    ).astype(int)

    metrics = compute_classification_metrics(
        y_test,
        y_pred,
        "Random Forest",
        probability_up,
    )

    metrics["best_params"] = (
        best_params
    )

    metrics["decision_threshold"] = (
        0.5
    )

    metrics["model_type"] = (
        "Random Forest Binary Trend Classifier"
    )

    pred_df = pd.DataFrame(
        {
            "Date": dates_test,
            "Actual": y_test,
            "Predicted": y_pred,
            "Probability_Uptrend": probability_up,
        }
    )

    pred_df = add_trend_columns(
        pred_df
    )

    save_prediction_and_metrics(
        model=model,
        metrics=metrics,
        pred_df=pred_df,
        save_dir=save_dir,
        model_filename="rf_model.pkl",
        prediction_filename="rf_predictions.csv",
        metrics_filename="rf_metrics.json",
    )

    return (
        model,
        metrics,
        pred_df,
    )


# Train XGBoost
def train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test,
    dates_test,
    tune=False,
    save_dir="models",
):
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    y_train = np.asarray(
        y_train
    ).astype(int).flatten()

    y_test = np.asarray(
        y_test
    ).astype(int).flatten()

    if tune:
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )

        tscv = TimeSeriesSplit(
            n_splits=3
        )

        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=tscv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(
            X_train,
            y_train,
        )

        model = (
            grid_search.best_estimator_
        )

        best_params = (
            grid_search.best_params_
        )

    else:
        positive_count = int(
            np.sum(y_train == 1)
        )

        negative_count = int(
            np.sum(y_train == 0)
        )

        scale_pos_weight = (
            negative_count
            / positive_count
            if positive_count > 0
            else 1.0
        )

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        best_params = {
            "scale_pos_weight": (
                scale_pos_weight
            )
        }

    probability_up = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    y_pred = (
        probability_up >= 0.5
    ).astype(int)

    metrics = compute_classification_metrics(
        y_test,
        y_pred,
        "XGBoost",
        probability_up,
    )

    metrics["best_params"] = (
        best_params
    )

    metrics["decision_threshold"] = (
        0.5
    )

    metrics["model_type"] = (
        "XGBoost Binary Trend Classifier"
    )

    pred_df = pd.DataFrame(
        {
            "Date": dates_test,
            "Actual": y_test,
            "Predicted": y_pred,
            "Probability_Uptrend": probability_up,
        }
    )

    pred_df = add_trend_columns(
        pred_df
    )

    save_prediction_and_metrics(
        model=model,
        metrics=metrics,
        pred_df=pred_df,
        save_dir=save_dir,
        model_filename="xgb_model.pkl",
        prediction_filename="xgb_predictions.csv",
        metrics_filename="xgb_metrics.json",
    )

    return (
        model,
        metrics,
        pred_df,
    )


# Train MLP(Multi Layer Preceptron)
def train_mlp(
    X_train,
    X_test,
    y_train,
    y_test,
    dates_test,
    save_dir="models",
    epochs=50,
    batch_size=16,
):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler

    os.makedirs(save_dir, exist_ok=True)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train).astype(int).flatten()
    y_test = np.asarray(y_test).astype(int).flatten()

    val_size = max(10, int(len(X_train) * 0.10))

    X_train_final = X_train[:-val_size]
    X_val = X_train[-val_size:]
    y_train_final = y_train[:-val_size]
    y_val = y_train[-val_size:]

    # Scaling hanya di-fit pada training-final.
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_final)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    model = Sequential(
        [
            Input(shape=(X_train_scaled.shape[1],)),

            Dense(
                16,
                activation="relu",
            ),
            BatchNormalization(),
            Dropout(0.35),

            Dense(
                16,
                activation="relu",
            ),
            Dropout(0.25),

            Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.001,
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=8,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train_scaled,
        y_train_final,
        validation_data=(
            X_val_scaled,
            y_val,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )

    val_probability_up = model.predict(
        X_val_scaled,
        verbose=0,
    ).flatten()

    decision_threshold = 0.50
    best_balanced_accuracy = -1.0
    best_f1 = -1.0

    for threshold in np.arange(0.30, 0.701, 0.01):
        val_pred = (
            val_probability_up >= threshold
        ).astype(int)

        balanced_acc = balanced_accuracy_score(
            y_val,
            val_pred,
        )

        current_f1 = f1_score(
            y_val,
            val_pred,
            zero_division=0,
        )

        if (
            balanced_acc > best_balanced_accuracy
            or (
                np.isclose(
                    balanced_acc,
                    best_balanced_accuracy,
                )
                and current_f1 > best_f1
            )
        ):
            decision_threshold = float(threshold)
            best_balanced_accuracy = float(balanced_acc)
            best_f1 = float(current_f1)

    probability_up = model.predict(
        X_test_scaled,
        verbose=0,
    ).flatten()

    y_pred = (
        probability_up >= decision_threshold
    ).astype(int)

    metrics = compute_classification_metrics(
        y_test,
        y_pred,
        "MLP",
        probability_up,
    )

    metrics["best_params"] = {
        "hidden_units_1": 32,
        "hidden_units_2": 16,
        "dropout_1": 0.35,
        "dropout_2": 0.25,
        "epochs_requested": epochs,
        "epochs_trained": len(history.history["loss"]),
        "batch_size": batch_size,
        "learning_rate": 0.001,
    }

    metrics["decision_threshold"] = round(
        float(decision_threshold),
        4,
    )
    metrics["validation_balanced_accuracy"] = round(
        float(best_balanced_accuracy * 100),
        4,
    )
    metrics["validation_f1"] = round(
        float(best_f1 * 100),
        4,
    )
    metrics["model_type"] = (
        "Dense Neural Network Binary Trend Classifier"
    )

    pred_df = pd.DataFrame(
        {
            "Date": dates_test,
            "Actual": y_test,
            "Predicted": y_pred,
            "Probability_Uptrend": probability_up,
        }
    )

    pred_df = add_trend_columns(pred_df)

    model.save(
        os.path.join(
            save_dir,
            "mlp_model.keras",
        )
    )

    joblib.dump(
        scaler_X,
        os.path.join(
            save_dir,
            "mlp_scaler_X.pkl",
        ),
    )

    pred_df.to_csv(
        os.path.join(
            save_dir,
            "mlp_predictions.csv",
        ),
        index=False,
    )

    pd.DataFrame(
        history.history
    ).to_csv(
        os.path.join(
            save_dir,
            "mlp_history.csv",
        ),
        index=False,
    )

    with open(
        os.path.join(
            save_dir,
            "mlp_metrics.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    return (
        model,
        metrics,
        pred_df,
    )