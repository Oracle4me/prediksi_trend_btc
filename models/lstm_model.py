import os
import json
import random

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from tensorflow.keras.callbacks import Callback


# Mengurangi log TensorFlow yang tidak penting.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Label binary:
# 0 = Turun / tidak naik
# 1 = Naik
TREND_LABELS = {
    0: "Turun",
    1: "Naik",
}


# Menyimpan bobot terbaik berdasarkan F1 validation.
class ValidationF1Checkpoint(Callback):
    def __init__(self, X_val, y_val, threshold=0.5):
        super().__init__()

        self.X_val = X_val
        self.y_val = np.asarray(y_val).astype(int).flatten()
        self.threshold = float(threshold)

        self.best_f1 = -1.0
        self.best_balanced_accuracy = -1.0
        self.best_accuracy = -1.0
        self.best_weights = None

    # Mengevaluasi model pada validation set setiap akhir epoch.
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_prob = self.model.predict(
            self.X_val,
            verbose=0,
        ).flatten()

        val_pred = (
            val_prob >= self.threshold
        ).astype(int)

        val_accuracy = float(
            accuracy_score(
                self.y_val,
                val_pred,
            )
        )

        val_balanced_accuracy = float(
            balanced_accuracy_score(
                self.y_val,
                val_pred,
            )
        )

        val_f1 = float(
            f1_score(
                self.y_val,
                val_pred,
                zero_division=0,
            )
        )

        logs["val_f1"] = val_f1
        logs["val_balanced_accuracy"] = val_balanced_accuracy

        is_better = (
            val_f1 > self.best_f1
            or (
                np.isclose(val_f1, self.best_f1)
                and val_balanced_accuracy > self.best_balanced_accuracy
            )
            or (
                np.isclose(val_f1, self.best_f1)
                and np.isclose(
                    val_balanced_accuracy,
                    self.best_balanced_accuracy,
                )
                and val_accuracy > self.best_accuracy
            )
        )

        if is_better:
            self.best_f1 = val_f1
            self.best_balanced_accuracy = val_balanced_accuracy
            self.best_accuracy = val_accuracy
            self.best_weights = self.model.get_weights()

    # Mengembalikan bobot terbaik setelah training selesai.
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(
                self.best_weights
            )


# Membuat hasil training lebih konsisten.
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    except Exception:
        pass


# Membagi training set menjadi train-final dan validation secara berurutan.
def split_train_validation(
    X_train,
    y_train,
    val_size=0.1,
):
    val_len = int(
        len(X_train) * val_size
    )

    if val_len < 10:
        raise ValueError(
            "Data training terlalu sedikit untuk validation set."
        )

    X_train_final = X_train[:-val_len]
    X_val = X_train[-val_len:]

    y_train_final = y_train[:-val_len]
    y_val = y_train[-val_len:]

    return (
        X_train_final,
        X_val,
        y_train_final,
        y_val,
    )


# Menghitung metric klasifikasi binary.
def compute_classification_metrics(
    y_actual,
    y_pred,
    name,
    y_prob=None,
):
    y_actual = (
        np.asarray(y_actual)
        .astype(int)
        .flatten()
    )

    y_pred = (
        np.asarray(y_pred)
        .astype(int)
        .flatten()
    )

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
    }

    if y_prob is not None:
        y_prob = (
            np.asarray(y_prob)
            .astype(float)
            .flatten()
        )

        metrics["Avg_Probability_Uptrend"] = round(
            float(
                np.mean(y_prob)
            ),
            4,
        )

    return metrics


# Menambahkan label trend teks ke dataframe prediksi.
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


# Menghitung bobot kelas binary dari distribusi training.
def make_class_weight_dict(y):
    y = (
        np.asarray(y)
        .astype(int)
        .flatten()
    )

    counts = np.bincount(
        y,
        minlength=2,
    )

    total = counts.sum()

    if (
        total == 0
        or counts[0] == 0
        or counts[1] == 0
    ):
        return None

    weight_0 = total / (
        2.0 * counts[0]
    )

    weight_1 = total / (
        2.0 * counts[1]
    )

    weight_0 = float(
        np.clip(
            weight_0,
            0.70,
            1.80,
        )
    )

    weight_1 = float(
        np.clip(
            weight_1,
            0.70,
            1.80,
        )
    )

    return {
        0: weight_0,
        1: weight_1,
    }


# Mencari threshold terbaik hanya dari validation set.
def find_best_threshold(
    y_true,
    probabilities,
    objective="accuracy",
):
    y_true = (
        np.asarray(y_true)
        .astype(int)
        .flatten()
    )

    probabilities = (
        np.asarray(probabilities)
        .astype(float)
        .flatten()
    )

    candidates = np.arange(
        0.20,
        0.801,
        0.01,
    )

    best_threshold = 0.5
    best_score = -1.0
    best_accuracy = -1.0
    best_balanced_accuracy = -1.0
    best_f1 = -1.0

    for threshold in candidates:
        prediction = (
            probabilities >= threshold
        ).astype(int)

        accuracy = float(
            accuracy_score(
                y_true,
                prediction,
            ) * 100
        )

        balanced_accuracy = float(
            balanced_accuracy_score(
                y_true,
                prediction,
            ) * 100
        )

        f1 = float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            ) * 100
        )

        if objective == "balanced":
            score = balanced_accuracy
        elif objective == "f1":
            score = f1
        else:
            score = accuracy

        is_better = (
            score > best_score
            or (
                np.isclose(
                    score,
                    best_score,
                )
                and balanced_accuracy
                > best_balanced_accuracy
            )
        )

        if is_better:
            best_threshold = float(
                threshold
            )
            best_score = score
            best_accuracy = accuracy
            best_balanced_accuracy = balanced_accuracy
            best_f1 = f1

    return (
        best_threshold,
        best_accuracy,
        best_balanced_accuracy,
        best_f1,
    )


# Menghitung baseline kelas mayoritas.
def majority_baseline_accuracy(y):
    y = (
        np.asarray(y)
        .astype(int)
        .flatten()
    )

    if len(y) == 0:
        return 0.0

    counts = np.bincount(
        y,
        minlength=2,
    )

    return round(
        float(
            np.max(counts)
            / len(y)
            * 100
        ),
        4,
    )


# Membuat arsitektur LSTM binary classifier.
def build_lstm_classifier(
    input_shape,
    units_1=96,
    units_2=48,
    dropout=0.25,
    learning_rate=0.0005,
    conv_filters=32,
    dense_units=48,
):
    import tensorflow as tf

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input,
        LSTM,
        Dense,
        Dropout,
        LayerNormalization,
        Conv1D,
        SpatialDropout1D,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    layers = [
        Input(
            shape=input_shape
        ),
        LayerNormalization(),
    ]

    if (
        conv_filters is not None
        and conv_filters > 0
    ):
        layers.extend(
            [
                Conv1D(
                    filters=conv_filters,
                    kernel_size=3,
                    padding="causal",
                    activation="relu",
                    kernel_regularizer=l2(
                        1e-5
                    ),
                ),

                LayerNormalization(),

                SpatialDropout1D(
                    dropout / 2
                ),
            ]
        )

    layers.extend(
        [
            LSTM(
                units_1,
                return_sequences=True,
                dropout=dropout / 2,
                recurrent_dropout=0.0,
                kernel_regularizer=l2(
                    1e-5
                ),
            ),

            LayerNormalization(),

            Dropout(
                dropout
            ),

            LSTM(
                units_2,
                return_sequences=False,
                dropout=dropout / 2,
                recurrent_dropout=0.0,
                kernel_regularizer=l2(
                    1e-5
                ),
            ),

            LayerNormalization(),

            Dropout(
                dropout
            ),

            Dense(
                dense_units,
                activation="relu",
                kernel_regularizer=l2(
                    1e-5
                ),
            ),

            Dropout(
                dropout / 2
            ),

            Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    model = Sequential(
        layers
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),

        loss=tf.keras.losses.BinaryCrossentropy(),

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
                threshold=0.5,
            ),
        ],
    )

    return model


# Melatih satu konfigurasi model LSTM.
def train_single_config(
    X_train,
    y_train,
    X_val,
    y_val,
    input_shape,
    config,
    epochs,
    seed=42,
    verbose=1,
):
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        TerminateOnNaN,
    )

    set_seed(seed)

    model = build_lstm_classifier(
        input_shape=input_shape,
        units_1=config["units_1"],
        units_2=config["units_2"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"],
        conv_filters=config.get(
            "conv_filters",
            32,
        ),
        dense_units=config.get(
            "dense_units",
            48,
        ),
    )

    f1_checkpoint = ValidationF1Checkpoint(
        X_val,
        y_val,
        threshold=0.5,
    )

    callbacks = [
        f1_checkpoint,

        EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=config.get(
                "patience",
                20,
            ),
            restore_best_weights=False,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.get(
                "lr_patience",
                6,
            ),
            min_lr=1e-6,
            verbose=0,
        ),

        TerminateOnNaN(),
    ]

    class_weight = (
        make_class_weight_dict(
            y_train
        )
        if config.get(
            "use_class_weight",
            True,
        )
        else None
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=config["batch_size"],
        validation_data=(
            X_val,
            y_val,
        ),
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=False,
        verbose=verbose,
    )

    val_loss = float(
        min(
            history.history["val_loss"]
        )
    )

    keras_val_accuracy = float(
        max(
            history.history["val_accuracy"]
        )
    )

    return (
        model,
        history,
        val_loss,
        keras_val_accuracy,
        class_weight,
    )


# Melatih dan mengevaluasi LSTM binary.
def train_lstm(
    X_train,
    X_test,
    y_train,
    y_test,
    dates_test,
    epochs=100,
    batch_size=16,
    save_dir="models",
    tune=True,
    val_size=0.1,
    threshold_objective="accuracy",
    use_ensemble=False,
    ensemble_top_k=1,
):
    set_seed(42)

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    X_train = np.asarray(
        X_train
    )

    X_test = np.asarray(
        X_test
    )

    y_train = (
        np.asarray(y_train)
        .astype(int)
        .flatten()
    )

    y_test = (
        np.asarray(y_test)
        .astype(int)
        .flatten()
    )

    input_shape = (
        X_train.shape[1],
        X_train.shape[2],
    )

    (
        X_train_final,
        X_val,
        y_train_final,
        y_val,
    ) = split_train_validation(
        X_train,
        y_train,
        val_size=val_size,
    )

    if tune:
        configs = [
            {
                "name": "ConvLSTM_96_48_do025_lr0005_bs16",
                "units_1": 96,
                "units_2": 48,
                "conv_filters": 32,
                "dense_units": 48,
                "dropout": 0.25,
                "learning_rate": 0.0005,
                "batch_size": batch_size,
                "patience": 20,
                "lr_patience": 6,
                "use_class_weight": True,
            },

            {
                "name": "LSTM_64_32_do020_lr0003_bs16",
                "units_1": 64,
                "units_2": 32,
                "conv_filters": 0,
                "dense_units": 32,
                "dropout": 0.20,
                "learning_rate": 0.0003,
                "batch_size": batch_size,
                "patience": 20,
                "lr_patience": 6,
                "use_class_weight": True,
            },
        ]

    else:
        configs = [
            {
                "name": "LSTM_Binary_Default",
                "units_1": 64,
                "units_2": 32,
                "conv_filters": 0,
                "dense_units": 32,
                "dropout": 0.20,
                "learning_rate": 0.0003,
                "batch_size": batch_size,
                "patience": 20,
                "lr_patience": 6,
                "use_class_weight": True,
            }
        ]

    seeds = (
        [42, 7, 123]
        if use_ensemble
        else [42]
    )

    trained_candidates = []
    tuning_results = []

    best_model = None
    best_history = None
    best_config = None
    best_seed = None

    best_val_loss = float("inf")
    best_val_accuracy = -1.0
    best_val_balanced_accuracy = -1.0
    best_val_f1 = -1.0
    best_threshold = 0.5

    total_runs = (
        len(configs)
        * len(seeds)
    )

    run_no = 0

    for config in configs:
        for seed in seeds:
            run_no += 1

            print(
                f"\n      Training LSTM "
                f"{run_no}/{total_runs}: "
                f"{config['name']} | "
                f"seed={seed}"
            )

            (
                model,
                history,
                val_loss,
                keras_val_accuracy,
                class_weight,
            ) = train_single_config(
                X_train=X_train_final,
                y_train=y_train_final,
                X_val=X_val,
                y_val=y_val,
                input_shape=input_shape,
                config=config,
                epochs=epochs,
                seed=seed,
                verbose=1,
            )

            val_prob = model.predict(
                X_val,
                verbose=0,
            ).flatten()

            (
                threshold,
                val_accuracy,
                val_balanced_accuracy,
                val_f1,
            ) = find_best_threshold(
                y_val,
                val_prob,
                objective=threshold_objective,
            )

            candidate = {
                "model": model,
                "history": history,
                "config": config,
                "seed": seed,
                "val_prob": val_prob,
                "threshold": threshold,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy": val_balanced_accuracy,
                "val_f1": val_f1,
                "keras_val_accuracy": keras_val_accuracy,
                "class_weight": class_weight,
            }

            trained_candidates.append(
                candidate
            )

            tuning_results.append(
                {
                    "config": config["name"],
                    "seed": seed,
                    "units_1": config["units_1"],
                    "units_2": config["units_2"],
                    "conv_filters": config.get(
                        "conv_filters",
                        0,
                    ),
                    "dense_units": config.get(
                        "dense_units",
                        0,
                    ),
                    "dropout": config["dropout"],
                    "learning_rate": config["learning_rate"],
                    "batch_size": config["batch_size"],
                    "class_weight": json.dumps(
                        class_weight
                    ),
                    "best_val_loss": round(
                        float(val_loss),
                        6,
                    ),
                    "keras_val_accuracy": round(
                        float(
                            keras_val_accuracy
                            * 100
                        ),
                        4,
                    ),
                    "decision_threshold": round(
                        float(threshold),
                        4,
                    ),
                    "val_trend_accuracy": round(
                        float(val_accuracy),
                        4,
                    ),
                    "val_balanced_accuracy": round(
                        float(
                            val_balanced_accuracy
                        ),
                        4,
                    ),
                    "val_f1": round(
                        float(val_f1),
                        4,
                    ),
                }
            )

            print(
                f"      Val Loss: "
                f"{val_loss:.6f} | "
                f"Threshold: "
                f"{threshold:.3f} | "
                f"Val Accuracy: "
                f"{val_accuracy:.2f}% | "
                f"Val BalancedAcc: "
                f"{val_balanced_accuracy:.2f}% | "
                f"Val F1: "
                f"{val_f1:.2f}%"
            )

            if threshold_objective == "balanced":
                current_score = (
                    val_balanced_accuracy
                )
                best_score = (
                    best_val_balanced_accuracy
                )

            elif threshold_objective == "f1":
                current_score = val_f1
                best_score = best_val_f1

            else:
                current_score = val_accuracy
                best_score = best_val_accuracy

            better_candidate = (
                best_model is None
                or current_score > best_score
                or (
                    np.isclose(
                        current_score,
                        best_score,
                    )
                    and val_f1 > best_val_f1
                )
                or (
                    np.isclose(
                        current_score,
                        best_score,
                    )
                    and np.isclose(
                        val_f1,
                        best_val_f1,
                    )
                    and val_loss
                    < best_val_loss
                )
            )

            if better_candidate:
                best_model = model
                best_history = history
                best_config = config
                best_seed = seed

                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_val_balanced_accuracy = (
                    val_balanced_accuracy
                )
                best_val_f1 = val_f1
                best_threshold = threshold

    trained_candidates = sorted(
        trained_candidates,
        key=lambda item: (
            item["val_accuracy"],
            item["val_f1"],
            item["val_balanced_accuracy"],
            -item["val_loss"],
        ),
        reverse=True,
    )

    if (
        use_ensemble
        and len(trained_candidates) > 1
    ):
        top_models = trained_candidates[
            : min(
                ensemble_top_k,
                len(trained_candidates),
            )
        ]

        ensemble_val_prob = np.mean(
            [
                item["val_prob"]
                for item in top_models
            ],
            axis=0,
        )

        (
            best_threshold,
            best_val_accuracy,
            best_val_balanced_accuracy,
            best_val_f1,
        ) = find_best_threshold(
            y_val,
            ensemble_val_prob,
            objective=threshold_objective,
        )

        test_prob = np.mean(
            [
                item["model"].predict(
                    X_test,
                    verbose=0,
                ).flatten()
                for item in top_models
            ],
            axis=0,
        )

        best_model = top_models[0]["model"]
        best_history = top_models[0]["history"]
        best_seed = None

        best_config = {
            "name": "ENSEMBLE_TOP_K_BINARY",
            "members": [
                {
                    "config": item[
                        "config"
                    ]["name"],
                    "seed": item["seed"],
                    "val_accuracy": item[
                        "val_accuracy"
                    ],
                    "val_f1": item[
                        "val_f1"
                    ],
                }
                for item in top_models
            ],
        }

        for rank, item in enumerate(
            top_models,
            start=1,
        ):
            item["model"].save(
                os.path.join(
                    save_dir,
                    f"lstm_ensemble_member_"
                    f"{rank}.keras",
                )
            )

    else:
        test_prob = best_model.predict(
            X_test,
            verbose=0,
        ).flatten()

    y_pred = (
        test_prob >= best_threshold
    ).astype(int)

    metrics = compute_classification_metrics(
        y_test,
        y_pred,
        "LSTM",
        test_prob,
    )

    metrics["best_config"] = (
        best_config["name"]
    )

    metrics["best_config_detail"] = (
        best_config
    )

    metrics["best_seed"] = best_seed

    metrics["best_val_loss"] = round(
        float(best_val_loss),
        6,
    )

    metrics["best_val_trend_accuracy"] = round(
        float(best_val_accuracy),
        4,
    )

    metrics[
        "best_val_balanced_accuracy"
    ] = round(
        float(
            best_val_balanced_accuracy
        ),
        4,
    )

    metrics["best_val_f1"] = round(
        float(best_val_f1),
        4,
    )

    metrics["decision_threshold"] = round(
        float(best_threshold),
        4,
    )

    metrics["threshold_objective"] = (
        threshold_objective
    )

    metrics["model_type"] = (
        "Conv1D + LSTM Binary Trend Classifier"
    )

    metrics["trend_classes"] = (
        TREND_LABELS
    )

    metrics["test_majority_baseline"] = (
        majority_baseline_accuracy(
            y_test
        )
    )

    metrics["val_majority_baseline"] = (
        majority_baseline_accuracy(
            y_val
        )
    )

    metrics["ensemble_enabled"] = bool(
        use_ensemble
    )

    metrics["evaluation_mode"] = (
        "pure_validation_selection"
    )

    metrics[
        "test_used_for_model_selection"
    ] = False

    best_model.save(
        os.path.join(
            save_dir,
            "lstm_model.keras",
        )
    )

    pred_df = pd.DataFrame(
        {
            "Date": dates_test,
            "Actual": y_test,
            "Predicted": y_pred,
            "Probability_Uptrend": test_prob,
        }
    )

    pred_df = add_trend_columns(
        pred_df
    )

    pred_df.to_csv(
        os.path.join(
            save_dir,
            "lstm_predictions.csv",
        ),
        index=False,
    )

    history_df = pd.DataFrame(
        best_history.history
    )

    history_df.to_csv(
        os.path.join(
            save_dir,
            "lstm_history.csv",
        ),
        index=False,
    )

    tuning_df = pd.DataFrame(
        tuning_results
    )

    tuning_df.to_csv(
        os.path.join(
            save_dir,
            "lstm_tuning_results.csv",
        ),
        index=False,
    )

    with open(
        os.path.join(
            save_dir,
            "lstm_metrics.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        "\n      LSTM Final Test Result"
    )

    print(
        f"      Balanced Accuracy    : "
        f"{metrics.get('Balanced_Accuracy', 0):.2f}%"
    )

    print(
        f"      Precision            : "
        f"{metrics.get('Precision', 0):.2f}%"
    )

    print(
        f"      Recall               : "
        f"{metrics.get('Recall', 0):.2f}%"
    )

    print(
        f"      F1 Score             : "
        f"{metrics.get('F1_Score', 0):.2f}%"
    )

    print(
        f"      Decision Threshold   : "
        f"{metrics.get('decision_threshold', 0):.3f}"
    )

    print(f"      Test Majority Base   : "f"{metrics.get('test_majority_baseline', 0):.2f}%")

    return (
        best_model,
        metrics,
        pred_df,
        best_history.history,
    )