import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler


def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def train_lstm_model(X_train, y_train, X_test, y_test, time_steps=20):

    # Fix: scale features — fit on train only, apply to both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create sequences AFTER split and scaling
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)

    # Fix: use a proper validation split from training data (not the test set)
    val_split = int(len(X_train_seq) * 0.8)
    X_val_seq = X_train_seq[val_split:]
    y_val_seq = y_train_seq[val_split:]
    X_tr_seq = X_train_seq[:val_split]
    y_tr_seq = y_train_seq[:val_split]

    # One-hot encoding
    y_tr_cat = to_categorical(y_tr_seq, num_classes=8)
    y_val_cat = to_categorical(y_val_seq, num_classes=8)

    # Compute class weights on the actual training portion
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_tr_seq),
        y=y_tr_seq
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Fix: right-sized model for 15 features (was 1024 units — overkill)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, X_train.shape[1])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(8, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("\nTraining LSTM...")
    model.fit(
        X_tr_seq,
        y_tr_cat,
        epochs=20,
        batch_size=256,
        validation_data=(X_val_seq, y_val_cat),
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=1
    )

    return model, scaler, X_test_seq, y_test_seq
