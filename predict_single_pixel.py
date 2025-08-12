# --- IMPORTS AND CONFIGURATION ---
import os
import h5py
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Configuration ---
# This script assumes it is being run from the '4_Inference_and_Visualization/' directory.
BASE_PROJECT_DIR = ".."
RAW_DATA_PATH = os.path.join(BASE_PROJECT_DIR, "1_Raw_Data", "SpatialData.mat")
ARTIFACTS_DIR = os.path.join(BASE_PROJECT_DIR, "2_Trained_Model_Artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "FINAL_CNN_FOCAL_LOSS_MODEL.h5")
ARTIFACTS_PATH = os.path.join(ARTIFACTS_DIR, "FINAL_DL_ARTIFACTS.joblib")

# === USER INPUTS: CONFIGURE THE PIXEL YOU WANT TO TEST ===
CASE_TO_TEST = 1          # Example: The case number (e.g., 'Case_5')
X_COORD_TO_TEST = 1     # Example: The X-coordinate of the pixel
Y_COORD_TO_TEST = 1      # Example: The Y-coordinate of the pixel

# Define where to save the output Excel file
OUTPUT_EXCEL_PATH = os.path.join(BASE_PROJECT_DIR, "4_Inference_and_Visualization", f"Pixel_Report_Case{CASE_TO_TEST}_({X_COORD_TO_TEST},{Y_COORD_TO_TEST}).xlsx")
# ==========================================================

# Constants that must match the training script
POSITIVE_CLASSES = [1, 3] # Cancer and DCIS
CLASS_NAMES_MAPPED = {0: 'Air', 1: 'Cancer', 2: 'Benign', 3: 'DCIS'}
BINARY_CLASS_NAMES = {0: 'Benign', 1: 'Cancer'}
Q_VALUES = np.arange(0.05, 0.501, 0.005)

# --- DATA EXTRACTION AND PREDICTION ---

def load_single_pixel_data(path, case_id, x_coord, y_coord):
    """Loads the 94-page data vector for a single specified pixel."""
    print(f"--- Loading data for Case_{case_id} at ({x_coord}, {y_coord}) ---")
    try:
        with h5py.File(path, 'r') as file:
            case_name = f"Case_{case_id}"
            if case_name not in file:
                print(f"FATAL: {case_name} not found in the data file.")
                return None

            data_cube = file[case_name]
            # Check if coordinates are valid
            if y_coord >= data_cube.shape[1] or x_coord >= data_cube.shape[0]:
                 print(f"FATAL: Coordinates ({x_coord}, {y_coord}) are out of bounds for this case. Max is ({data_cube.shape[0]-1}, {data_cube.shape[1]-1}).")
                 return None

            # h5py reads in (pages, y, x) format, but our model expects (x, y, pages)
            # We must read the vector correctly. The data was saved as (x,y,pages)
            pixel_vector = data_cube[x_coord, y_coord, :]

            true_label_id = int(pixel_vector[0])
            if true_label_id == 0:
                print("WARNING: Selected pixel is 'Air'. Model is not trained on this.")

            # Extract spectral and spatial features
            spatial_features = np.array([pixel_vector[1]]) # TSI
            # Corrected indexing to get the last 91 values as spectral features
            spectral_features = pixel_vector[-len(Q_VALUES):]

            return spectral_features, spatial_features, true_label_id

    except Exception as e:
        print(f"FATAL: An error occurred while loading data: {e}")
        return None

def main():
    """Main function to load model, predict on a single pixel, and save to Excel."""
    print("--- Starting Single Pixel Inference Script ---")

    # --- Step 1: Load all necessary artifacts ---
    print("\n--- Loading final model and artifacts ---")
    try:
        def focal_loss(gamma=2., alpha=.25):
            def focal_loss_fixed(y_true, y_pred):
                # Custom object must be defined to load the model
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

    # --- Step 2: Extract the single datapoint ---
    pixel_data = load_single_pixel_data(RAW_DATA_PATH, CASE_TO_TEST, X_COORD_TO_TEST, Y_COORD_TO_TEST)
    if pixel_data is None: return
    spectral_features, spatial_features, true_label_id = pixel_data

    # --- Step 3: Normalize the datapoint using loaded scalers ---
    spectral_s = artifacts['spectral_scaler'].transform([spectral_features])
    spatial_s = artifacts['spatial_scaler'].transform([spatial_features])

    # Reshape for the CNN model (add batch and feature dimensions)
    spectral_s_reshaped = np.expand_dims(spectral_s, axis=2)

    # --- Step 4: Run prediction ---
    print("\n--- Running prediction on the single pixel ---")
    predicted_proba = model.predict([spectral_s_reshaped, spatial_s], verbose=0)[0][0]

    # --- Step 5: Classify using the optimal threshold ---
    final_prediction = (predicted_proba >= artifacts['optimal_threshold']).astype(int)
    final_label_name = BINARY_CLASS_NAMES[final_prediction]

    true_label_name = CLASS_NAMES_MAPPED.get(true_label_id, "Unknown")

    print(f"...Predicted Probability of Cancer: {predicted_proba:.4f}")
    print(f"...Final Classification: ** {final_label_name} ** (True Label was: {true_label_name})")

    # --- Step 6: Create and save the Excel report ---
    print(f"\n--- Saving report to Excel file: {OUTPUT_EXCEL_PATH} ---")

    # Create the Summary Report DataFrame
    summary_data = {
        "Case ID": [CASE_TO_TEST],
        "X Coordinate": [X_COORD_TO_TEST],
        "Y Coordinate": [Y_COORD_TO_TEST],
        "True Label ID": [true_label_id],
        "True Label Name": [true_label_name],
        "Model Optimal Threshold": [f"{artifacts['optimal_threshold']:.4f}"],
        "Predicted Probability (Cancer)": [f"{predicted_proba:.4f}"],
        "Final Classification": [final_label_name]
    }
    summary_df = pd.DataFrame(summary_data)

    # Create the Full Spectrum DataFrame
    spectrum_df = pd.DataFrame({
        'q_value': Q_VALUES,
        'Intensity': spectral_features
    })

    # Use ExcelWriter to save to multiple sheets
    with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary_Report', index=False)
        spectrum_df.to_excel(writer, sheet_name='Full_Spectrum', index=False)

    print("...Excel report saved successfully.")
    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()
