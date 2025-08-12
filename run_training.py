# --- IMPORTS AND CONFIGURATION ---
import os
import h5py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Flatten,
                            Concatenate, Dropout, BatchNormalization)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Configuration ---
# This script assumes it is being run from the '3_Training_Script/' directory.
# Paths are relative to the project root.
BASE_PROJECT_DIR = ".." # Go up one level to the project root
RAW_DATA_PATH = os.path.join(BASE_PROJECT_DIR, "1_Raw_Data", "SpatialData.mat")
ARTIFACTS_SAVE_DIR = os.path.join(BASE_PROJECT_DIR, "2_Trained_Model_Artifacts")
MODEL_FILENAME = "FINAL_CNN_FOCAL_LOSS_MODEL.h5"
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_SAVE_DIR, MODEL_FILENAME)
ARTIFACTS_FILENAME = "FINAL_DL_ARTIFACTS.joblib"
ARTIFACTS_SAVE_PATH = os.path.join(ARTIFACTS_SAVE_DIR, ARTIFACTS_FILENAME)

RANDOM_STATE = 42
POSITIVE_CLASSES = [1, 3] # Cancer and DCIS
CLASS_NAMES_MAPPED = {0: 'Negative (Benign)', 1: 'Positive (Cancer/DCIS)'}
MINIMUM_TARGET_RECALL = 0.90

# --- DATA PREPARATION ---
def create_cnn_dataframe(path):
    print("\n--- Step 1: Loading Data for Deep Learning Model ---")
    try:
        with h5py.File(path, 'r') as all_case_data:
            list_of_dfs = []
            Q_VALUES = np.arange(0.05, 0.501, 0.005)
            sorted_case_names = sorted(all_case_data.keys(), key=lambda x: int(x.split('_')[1]))
            for case_name in sorted_case_names:
                print(f"  - Processing {case_name}...")
                case_data_3d = all_case_data[case_name][:].T
                xrd_data_flat = case_data_3d[:, :, 3:].reshape(-1, len(Q_VALUES))
                q_cols = [f'q_{i}' for i in range(xrd_data_flat.shape[1])]
                xrd_df = pd.DataFrame(xrd_data_flat, columns=q_cols)
                total_scatter_map = case_data_3d[:, :, 1]
                spatial_df = pd.DataFrame({'label_id': case_data_3d[:, :, 0].flatten(), 'total_scatter': total_scatter_map.flatten()})
                list_of_dfs.append(pd.concat([spatial_df, xrd_df], axis=1))
            master_df = pd.concat(list_of_dfs, ignore_index=True)
            print("--- ...Data Prep Complete ---")
            return master_df[master_df['label_id'] != 0].copy()
    except FileNotFoundError:
        print(f"FATAL: Raw data file not found at {path}")
        return None

# --- DEEP LEARNING MODEL DEFINITION ---
def build_final_1d_cnn_model(spectral_input_shape, spatial_input_shape):
    spectral_input = Input(shape=spectral_input_shape, name='spectral_input')
    x = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(spectral_input)
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(x)
    x = BatchNormalization()(x); x = MaxPooling1D(pool_size=2)(x); x = Dropout(0.3)(x)
    x = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x); x = MaxPooling1D(pool_size=2)(x); x = Dropout(0.4)(x)
    x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x); x = MaxPooling1D(pool_size=2)(x); x = Dropout(0.5)(x)
    x = Flatten()(x)
    spatial_input = Input(shape=spatial_input_shape, name='spatial_input')
    combined = Concatenate()([x, spatial_input])
    z = Dense(256, activation='relu')(combined); z = Dropout(0.5)(z)
    z = Dense(128, activation='relu')(z)
    output = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[spectral_input, spatial_input], outputs=output)
    
    def focal_loss(gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            epsilon = tf.keras.backend.epsilon()
            return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + epsilon)) \
                   -tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + epsilon))
        return focal_loss_fixed
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=focal_loss(), metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# --- MAIN WORKFLOW ---
def main():
    master_df = create_cnn_dataframe(RAW_DATA_PATH)
    if master_df is None: return

    master_df['label'] = master_df['label_id'].apply(lambda x: 1 if x in POSITIVE_CLASSES else 0)
    spectral_features = [c for c in master_df.columns if c.startswith('q_')]
    spatial_features = ['total_scatter']
    target = 'label'

    train_df, test_df = train_test_split(master_df, test_size=0.2, random_state=RANDOM_STATE, stratify=master_df[target])
    
    X_train_spectral, y_train = train_df[spectral_features].values, train_df[target].values
    X_train_spatial = train_df[spatial_features].values
    X_test_spectral, y_test = test_df[spectral_features].values, test_df[target].values
    X_test_spatial = test_df[spatial_features].values

    spectral_scaler = StandardScaler().fit(X_train_spectral)
    X_train_spectral_s = spectral_scaler.transform(X_train_spectral)
    X_test_spectral_s = spectral_scaler.transform(X_test_spectral)
    spatial_scaler = StandardScaler().fit(X_train_spatial)
    X_train_spatial_s = spatial_scaler.transform(X_train_spatial)
    X_test_spatial_s = spatial_scaler.transform(X_test_spatial)
    
    X_train_spectral_s = np.expand_dims(X_train_spectral_s, axis=2)
    X_test_spectral_s = np.expand_dims(X_test_spectral_s, axis=2)

    model = build_final_1d_cnn_model(spectral_input_shape=(len(spectral_features), 1),
                                     spatial_input_shape=(len(spatial_features),))
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.2, patience=5, min_lr=1e-6)

    print("\n--- Training Final Deep Learning Model ---")
    model.fit(
        {'spectral_input': X_train_spectral_s, 'spatial_input': X_train_spatial_s}, y_train,
        validation_split=0.2, epochs=150, batch_size=512,
        callbacks=[early_stopping, reduce_lr], verbose=1
    )

    final_probas = model.predict([X_test_spectral_s, X_test_spatial_s]).flatten()
    
    precision, recall, thresholds = precision_recall_curve(y_test, final_probas)
    valid_indices = np.where(recall[:-1] >= MINIMUM_TARGET_RECALL)[0]
    best_threshold = thresholds[valid_indices[np.argmax(precision[valid_indices])]] if len(valid_indices) > 0 else 0.5
    final_predictions = (final_probas >= best_threshold).astype(int)

    os.makedirs(ARTIFACTS_SAVE_DIR, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    joblib.dump({'spectral_scaler': spectral_scaler, 'spatial_scaler': spatial_scaler,
                 'optimal_threshold': best_threshold}, ARTIFACTS_SAVE_PATH)
    print(f"\n--- Model and artifacts saved to '{ARTIFACTS_SAVE_DIR}' ---")
    
    print("\n\n" + "="*80); print(" FINAL EVALUATION: FINAL OPTIMIZED DEEP LEARNING MODEL"); print("="*80)
    print(f"\nOverall Accuracy: {accuracy_score(y_test, final_predictions):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, final_probas):.4f}\n")
    print(f"Classification Report (at threshold for recall >= {MINIMUM_TARGET_RECALL}):")
    print(classification_report(y_test, final_predictions, target_names=CLASS_NAMES_MAPPED.values(), zero_division=0))
    cm = confusion_matrix(y_test, final_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES_MAPPED.values(), yticklabels=CLASS_NAMES_MAPPED.values())
    plt.title('Final Optimized DL Model Confusion Matrix'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.show()

if __name__ == '__main__':
    main()
