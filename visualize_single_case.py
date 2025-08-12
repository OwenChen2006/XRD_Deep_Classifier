# --- IMPORTS AND CONFIGURATION ---
import os
import h5py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from scipy.ndimage import gaussian_filter

# --- Configuration ---
# This script assumes it is being run from the '4_Inference_and_Visualization/' directory.
BASE_PROJECT_DIR = ".." # Go up one level to the project root
RAW_DATA_PATH = os.path.join(BASE_PROJECT_DIR, "1_Raw_Data", "SpatialData.mat")
ARTIFACTS_DIR = os.path.join(BASE_PROJECT_DIR, "2_Trained_Model_Artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "FINAL_CNN_FOCAL_LOSS_MODEL.h5")
ARTIFACTS_PATH = os.path.join(ARTIFACTS_DIR, "FINAL_DL_ARTIFACTS.joblib")
VISUALIZATION_OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "4_Inference_and_Visualization", "visualizations")

# --- The specific case you want to visualize ---
CASE_TO_ANALYZE = 'Case_1'

# Constants that must match the training script
POSITIVE_LABELS_ORIGINAL = [1, 3]
Q_VALUES = np.arange(0.05, 0.501, 0.005)

# --- DATA PREPARATION AND PREDICTION FUNCTIONS ---
def load_and_prep_case_for_dl(path, case_name):
    """Loads and prepares the data for a single case for CNN inference."""
    try:
        with h5py.File(path, 'r') as file:
            case_data_3d = file[case_name][:].T
    except Exception as e:
        print(f"[ERROR] Could not load raw data for {case_name}: {e}"); return None
    
    q_cols = [f'q_{i}' for i in range(len(Q_VALUES))]
    xrd_df = pd.DataFrame(case_data_3d[:, :, 3:].reshape(-1, len(Q_VALUES)), columns=q_cols)
    df = pd.DataFrame({
        'x_coord': np.repeat(np.arange(case_data_3d.shape[0]), case_data_3d.shape[1]),
        'y_coord': np.tile(np.arange(case_data_3d.shape[1]), case_data_3d.shape[0]),
        'label_id': case_data_3d[:, :, 0].flatten().astype(int),
        'total_scatter': case_data_3d[:, :, 1].flatten()})
    return pd.concat([df, xrd_df], axis=1)

def predict_case_with_dl_model(df, model, artifacts):
    """Uses the loaded model and artifacts to make predictions on a case DataFrame."""
    spectral_scaler = artifacts['spectral_scaler']
    spatial_scaler = artifacts['spatial_scaler']
    threshold = artifacts['optimal_threshold']
    tissue_df = df[df['label_id'] != 0].copy()
    if tissue_df.empty:
        df['classification_result'] = 'Air'
        return df

    X_spectral = tissue_df[[f'q_{i}' for i in range(len(Q_VALUES))]].values
    X_spatial = tissue_df[['total_scatter']].values
    X_spectral_s = spectral_scaler.transform(X_spectral)
    X_spatial_s = spatial_scaler.transform(X_spatial)
    X_spectral_s = np.expand_dims(X_spectral_s, axis=2)

    probas = model.predict([X_spectral_s, X_spatial_s], verbose=0).flatten()
    predictions = (probas >= threshold).astype(int)
    tissue_df['predicted_id'] = predictions

    conditions = [
        (tissue_df['label_id'].isin(POSITIVE_LABELS_ORIGINAL)) & (tissue_df['predicted_id'] == 1),
        (~tissue_df['label_id'].isin(POSITIVE_LABELS_ORIGINAL)) & (tissue_df['predicted_id'] == 0),
        (~tissue_df['label_id'].isin(POSITIVE_LABELS_ORIGINAL)) & (tissue_df['predicted_id'] == 1),
        (tissue_df['label_id'].isin(POSITIVE_LABELS_ORIGINAL)) & (tissue_df['predicted_id'] == 0)]
    results = ['TP', 'TN', 'FP', 'FN']
    tissue_df['classification_result'] = np.select(conditions, results, default='Unknown')

    df = df.merge(tissue_df[['classification_result']], left_index=True, right_index=True, how='left')
    df['classification_result'] = df['classification_result'].fillna('Air')
    return df

# --- PLOTTING FUNCTIONS ---
def plot_visual_maps(df, case_name, output_dir):
    print(f"  > Generating visual maps for {case_name}...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plt.suptitle(f'Visual Analysis for {case_name} (Final DL Model)', fontsize=20, weight='bold')
    max_x, max_y = df['x_coord'].max(), df['y_coord'].max()
    grid_shape = (max_x + 1, max_y + 1)
    q_cols = [c for c in df.columns if c.startswith('q_')]
    df['mean_q'] = df[q_cols].mean(axis=1)
    mean_q_grid = np.full(grid_shape, np.nan); mean_q_grid[df['x_coord'].values, df['y_coord'].values] = df['mean_q'].values
    ax1 = axes[0]; im1 = ax1.imshow(mean_q_grid, cmap='viridis', origin='lower')
    ax1.set_title('Mean q-value Intensity', fontsize=16); ax1.set_xticks([]); ax1.set_yticks([])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Mean q (1/Å)')
    category_map = {'Air': 0, 'TN': 1, 'TP': 2, 'FN': 3, 'FP': 4}
    color_map = {0: 'black', 1: '#808080', 2: '#FFA500', 3: '#800080', 4: '#00AEEF'}
    classification_grid = np.full(grid_shape, category_map['Air']); classification_grid[df['x_coord'].values, df['y_coord'].values] = df['classification_result'].map(category_map).values
    cmap = plt.cm.colors.ListedColormap([color_map[i] for i in sorted(color_map)])
    norm = plt.cm.colors.BoundaryNorm(np.arange(len(color_map) + 1) - 0.5, len(color_map))
    ax2 = axes[1]; ax2.imshow(classification_grid, cmap=cmap, norm=norm, origin='lower')
    ax2.set_title('Classification Results', fontsize=16); ax2.set_xticks([]); ax2.set_yticks([])
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in [('TP', '#FFA500'), ('TN', '#808080'), ('FP', '#00AEEF'), ('FN', '#800080'), ('Air', 'black')]]
    ax2.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    save_path = os.path.join(output_dir, f"{case_name}_DL_Visual_Maps.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"    ...Saved map plot to {save_path}")

def plot_spectra_comparison(df, case_name, output_dir):
    print(f"  > Generating spectra comparison for {case_name}...")
    q_cols = [c for c in df.columns if c.startswith('q_')]
    if not q_cols: return
    categories, titles, colors = ['TP', 'TN', 'FP', 'FN'], ['True Positives', 'True Negatives', 'False Positives', 'False Negatives'], ['#FFA500', '#808080', '#00AEEF', '#800080']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f'Normalized XRD Spectra Comparison for {case_name} (Final DL Model)', fontsize=20, weight='bold')
    for ax, category, title, color in zip(axes.flatten(), categories, titles, colors):
        subset_df = df[df['classification_result'] == category]
        ax.set_title(title, fontsize=16)
        if not subset_df.empty:
            spectra = subset_df[q_cols].values
            min_vals, max_vals = spectra.min(axis=1, keepdims=True), spectra.max(axis=1, keepdims=True)
            range_vals = max_vals - min_vals; range_vals[range_vals == 0] = 1
            normalized_spectra = (spectra - min_vals) / range_vals
            ax.plot(Q_VALUES, normalized_spectra.T, color=color, alpha=0.05)
            ax.plot(Q_VALUES, normalized_spectra.mean(axis=0), color='black', linestyle='--', linewidth=2.5)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, style='italic')
        ax.set_ylabel('Normalized Intensity'); ax.grid(True, linestyle=':'); ax.set_xlim([Q_VALUES.min(), 0.3]); ax.set_ylim([-0.05, 1.05])
    fig.text(0.5, 0.06, 'q (1/Å)', ha='center', fontsize=14)
    save_path = os.path.join(output_dir, f"{case_name}_DL_Spectra_Comparison.png")
    plt.tight_layout(rect=[0, 0.07, 1, 0.95]); plt.savefig(save_path, dpi=150); plt.close(fig)
    print(f"    ...Saved spectra plot to {save_path}")

# --- MAIN INFERENCE WORKFLOW ---
def main():
    print("--- Starting DL Model Diagnostic Visualization Script ---")
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

    print("\n--- Loading final model and artifacts ---")
    try:
        def focal_loss(gamma=2., alpha=.25): # Must redefine to load
            def focal_loss_fixed(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
                pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
                epsilon = tf.keras.backend.epsilon()
                return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + epsilon)) \
                       -tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + epsilon))
            return focal_loss_fixed
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})
        artifacts = joblib.load(ARTIFACTS_PATH)
        print("...Success! Model and artifacts loaded.")
    except Exception as e:
        print(f"[FATAL ERROR] Could not load files: {e}"); return
    
    print(f"\n{'='*25} Processing {CASE_TO_ANALYZE} {'='*25}")
    case_df = load_and_prep_case_for_dl(RAW_DATA_PATH, CASE_TO_ANALYZE)
    if case_df is None: return
    
    final_df = predict_case_with_dl_model(case_df, model, artifacts)
    
    plot_visual_maps(final_df, CASE_TO_ANALYZE, VISUALIZATION_OUTPUT_DIR)
    plot_spectra_comparison(final_df, CASE_TO_ANALYZE, VISUALIZATION_OUTPUT_DIR)

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()
