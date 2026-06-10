import os
import json
import numpy as np
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

TIME_STEPS = 20
N_CLASSES  = 8


def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def _make_model_builder(n_features):
    def build_model(hp):
        units_1         = hp.Choice('units_1',        [64, 128, 256])
        units_2         = hp.Choice('units_2',        [32, 64, 128])
        dropout         = hp.Float('dropout',          0.1, 0.5, step=0.1)
        lr              = hp.Choice('learning_rate',  [1e-4, 5e-4, 1e-3])
        dense_units     = hp.Choice('dense_units',    [32, 64, 128])
        num_lstm_layers = hp.Int('num_lstm_layers',    1, 3)
        bidirectional   = hp.Boolean('bidirectional')
        batch_norm      = hp.Boolean('batch_norm')
        l2_val          = hp.Choice('l2_reg',         [0.0, 1e-4, 1e-3])

        reg = l2(l2_val) if l2_val > 0 else None

        model = Sequential()

        # ── First LSTM layer (carries input_shape) ──────────────────────────
        first_lstm = LSTM(
            units_1,
            return_sequences=(num_lstm_layers > 1),
            kernel_regularizer=reg,
            input_shape=(TIME_STEPS, n_features),
        )
        model.add(Bidirectional(first_lstm) if bidirectional else first_lstm)
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout))

        # ── Extra LSTM layers ────────────────────────────────────────────────
        for i in range(1, num_lstm_layers):
            return_seq = (i < num_lstm_layers - 1)
            lstm_layer = LSTM(
                units_2,
                return_sequences=return_seq,
                kernel_regularizer=reg,
            )
            model.add(Bidirectional(lstm_layer) if bidirectional else lstm_layer)
            if batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))

        # ── Dense head ───────────────────────────────────────────────────────
        model.add(Dense(dense_units, activation='relu', kernel_regularizer=reg))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dense(N_CLASSES, activation='softmax'))

        model.compile(
            optimizer=Adam(lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model
    return build_model


def tune_and_train_lstm(X_train, y_train, X_test, y_test,
                        tuner_dir='results/tuner',
                        max_trials=15,
                        results_dir='results'):

    # ── Scale (fit on train only) ────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Sequences ────────────────────────────────────────────────────────────
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
    X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test)

    # ── Validation split (last 20% of train sequences) ───────────────────────
    val_split = int(len(X_train_seq) * 0.8)
    X_tr,  y_tr  = X_train_seq[:val_split], y_train_seq[:val_split]
    X_val, y_val = X_train_seq[val_split:], y_train_seq[val_split:]

    y_tr_cat  = to_categorical(y_tr,  num_classes=N_CLASSES)
    y_val_cat = to_categorical(y_val, num_classes=N_CLASSES)

    # ── Class weights ────────────────────────────────────────────────────────
    class_weights     = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weights_dict = dict(enumerate(class_weights))

    # ── Keras Tuner ──────────────────────────────────────────────────────────
    n_features = X_train.shape[1]
    builder    = _make_model_builder(n_features)

    tuner = kt.BayesianOptimization(
        builder,
        objective='val_accuracy',
        max_trials=max_trials,
        directory=tuner_dir,
        project_name='lstm_iot_ids',
        overwrite=True,
    )

    callbacks_search = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                          min_lr=1e-6, verbose=0),
    ]

    print(f"\nStarting Keras Tuner search ({max_trials} trials) ...")
    tuner.search(
        X_tr, y_tr_cat,
        epochs=20,
        batch_size=256,
        validation_data=(X_val, y_val_cat),
        class_weight=class_weights_dict,
        callbacks=callbacks_search,
        verbose=1,
    )

    # ── Print & save best hyperparameters ────────────────────────────────────
    best_hp = tuner.get_best_hyperparameters(1)[0]
    hp_keys = ['units_1', 'units_2', 'dropout', 'learning_rate',
               'dense_units', 'num_lstm_layers', 'bidirectional',
               'batch_norm', 'l2_reg']

    print("\n========== Best Hyperparameters ==========")
    hp_dict = {}
    for key in hp_keys:
        val = best_hp.get(key)
        hp_dict[key] = val
        print(f"  {key:20s}: {val}")
    print("==========================================")

    os.makedirs(results_dir, exist_ok=True)
    hp_path = os.path.join(results_dir, 'lstm_best_hyperparams.json')
    with open(hp_path, 'w') as f:
        json.dump(hp_dict, f, indent=2)
    print(f"Saved hyperparameters → {hp_path}")

    # ── Retrain best model on full training sequences ────────────────────────
    print("\nRetraining best model on full training data ...")
    callbacks_final = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                          min_lr=1e-6, verbose=1),
    ]

    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        X_train_seq, to_categorical(y_train_seq, num_classes=N_CLASSES),
        epochs=30,
        batch_size=256,
        validation_data=(X_val, y_val_cat),
        class_weight=class_weights_dict,
        callbacks=callbacks_final,
        verbose=1,
    )

    return best_model, scaler, X_test_seq, y_test_seq, best_hp
