# --- IMPORTS AND CONFIGURATION ---
import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import lightgbm as lgb
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# --- Configuration ---
# IMPORTANT: Update these paths to match your project structure.
BASE_PROJECT_DIR = r"C:\Path\To\Your\Main\Project\Folder"
ARTIFACTS_DIR = os.path.join(BASE_PROJECT_DIR, "Final_LGBM_Ratio_Model")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "FINAL_LGBM_MODEL.joblib")
ARTIFACTS_PATH = os.path.join(ARTIFACTS_DIR, "FINAL_LGBM_ARTIFACTS.joblib")

# Path to the new, raw data file you want to analyze
DATA_TO_ANALYZE = os.path.join(BASE_PROJECT_DIR, "New_Data_To_Classify", "45CM-10c_Recon.csv")

# A safe output directory on your Desktop
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "XRD_Inference_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants must match the training script
Q_VALUES = np.arange(0.05, 0.501, 0.005)

# --- FEATURE ENGINEERING (Must be identical to training) ---

def get_feature_names():
    """Returns a consistent list of final engineered feature names."""
    return ['primary_peak_loc', 'primary_peak_prom', 'peak2_loc_diff', 'peak2_height_ratio',
            'peak2_prom_ratio', 'peak3_loc_diff', 'peak3_height_ratio', 'peak3_prom_ratio',
            'num_peaks', 'tsi']

def analyze_peak_ratios(pixel_row_with_tsi):
    """
    Takes a single pixel row (91 spectral values + 1 TSI value) and
    engineers the final scale-invariant feature set.
    """
    xrd_intensities = pixel_row_with_tsi[:-1]
    tsi = pixel_row_with_tsi[-1]
    try:
        peaks, props = find_peaks(xrd_intensities, prominence=0.01, width=1, distance=3)
        if len(peaks) == 0: return pd.Series([0.0] * 10, index=get_feature_names())
    except Exception:
        return pd.Series([0.0] * 10, index=get_feature_names())
    heights = props.get('peak_heights', np.array([])); prominences = props.get('prominences', np.array([]))
    sorted_indices = np.argsort(heights)[::-1]
    top_peaks, top_heights, top_proms = peaks[sorted_indices[:3]], heights[sorted_indices[:3]], prominences[sorted_indices[:3]]
    features = {name: 0.0 for name in get_feature_names()}
    features['num_peaks'], features['tsi'] = float(len(peaks)), tsi
    if len(top_peaks) > 0:
        primary_loc = Q_VALUES[top_peaks[0]]
        primary_h = top_heights[0] or 1.0; primary_p = top_proms[0] or 1.0
        features['primary_peak_loc'], features['primary_peak_prom'] = primary_loc, primary_p
        if len(top_peaks) > 1:
            features.update({'peak2_loc_diff': Q_VALUES[top_peaks[1]]-primary_loc,
                             'peak2_height_ratio': top_heights[1]/primary_h if primary_h else 0,
                             'peak2_prom_ratio': top_proms[1]/primary_p if primary_p else 0})
        if len(top_peaks) > 2:
            features.update({'peak3_loc_diff': Q_VALUES[top_peaks[2]]-primary_loc,
                             'peak3_height_ratio': top_heights[2]/primary_h if primary_h else 0,
                             'peak3_prom_ratio': top_proms[2]/primary_p if primary_p else 0})
    return pd.Series(features)

def prepare_inference_data(path):
    """Loads a new case CSV, calculates TSI, and engineers features."""
    print(f"\n--- Step 1: Loading and Preparing Data from {os.path.basename(path)} ---")
    try:
        raw_data = pd.read_csv(path, header=None)
        pixel_data = raw_data.transpose()
        if pixel_data.shape[1] != 91:
            print(f"FATAL: Expected 91 spectral columns, but found {pixel_data.shape[1]}."); return None
        
        spectral_data = pixel_data.values
        tsi = spectral_data.sum(axis=1).reshape(-1, 1)
        data_for_engineering = np.hstack([spectral_data, tsi])
        
        print(f"  - Calculating TSI and engineering features for {len(data_for_engineering)} pixels...")
        features_df = pd.DataFrame(data_for_engineering).apply(analyze_peak_ratios, axis=1)
        
        print("--- ...Data Preparation Complete ---")
        return features_df
    except Exception as e:
        print(f"FATAL: Error during data preparation: {e}"); return None

def predict_with_ensemble(models, X_test):
    """Averages probabilities from all models in the ensemble."""
    if not models: return np.array([])
    all_probas = [model.predict_proba(X_test) for model in models]
    return np.mean(all_probas, axis=0)

# --- MAIN INFERENCE WORKFLOW ---
def main():
    """Main function to run the full inference pipeline."""
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    print("--- Starting XRD Cancer Classification Inference ---")

    # Step 1: Load all model artifacts
    print("\n--- Loading final model and artifacts ---")
    try:
        model = joblib.load(MODEL_PATH)
        artifacts = joblib.load(ARTIFACTS_PATH)
        scaler = artifacts['scaler']
        decision_threshold = artifacts['optimal_threshold']
        feature_names = artifacts['features']
        print("...Success! Model and artifacts loaded.")
    except Exception as e:
        print(f"FATAL: Could not load model or artifacts. Check paths in script. Error: {e}"); return

    # Step 2: Prepare the inference data
    features_df = prepare_inference_data(DATA_TO_ANALYZE)
    if features_df is None: return

    # Ensure the feature order is identical to training
    X_inference = features_df[feature_names].values
    
    # Step 3: Normalize the new data using the loaded scaler
    print("\n--- Step 2: Normalizing Data ---")
    X_inference_s = scaler.transform(X_inference)
    print("...Data normalized successfully.")

    # Step 4: Run prediction
    print("\n--- Step 3: Running Prediction ---")
    pixel_probas = predict_with_ensemble(model, X_inference_s)
    # Get probability for the "Positive" class (class 1)
    positive_probas = pixel_probas[:, 1]
    print("...Prediction complete.")

    # Step 5: Apply threshold and generate results
    pixel_preds = (positive_probas >= decision_threshold).astype(int)
    results_df = pd.DataFrame({
        'Pixel_Index': features_df.index + 1,
        'Cancer_Probability': positive_probas,
        'Final_Prediction': pixel_preds
    }).map({0: 'Benign', 1: 'Cancer'})
    
    # --- Step 6: Display and Save Report ---
    num_cancer_pixels = np.sum(pixel_preds)
    num_benign_pixels = len(pixel_preds) - num_cancer_pixels

    print("\n\n" + "="*60); print("        FINAL INFERENCE REPORT"); print("="*60)
    print(f"Source File: {os.path.basename(DATA_TO_ANALYZE)}")
    print(f"Total Pixels Analyzed: {len(pixel_preds)}")
    print(f"Using Decision Threshold: {decision_threshold:.4f}")
    print("-" * 60)
    print(f"Pixels Classified as 'Cancer': {num_cancer_pixels} ({num_cancer_pixels/len(pixel_preds):.2%})")
    print(f"Pixels Classified as 'Benign': {num_benign_pixels}")
    print("="*60)
    
    print("\n--- Detailed Per-Pixel Results ---")
    print(results_df.to_string())

    output_filename = f"prediction_{os.path.splitext(os.path.basename(DATA_TO_ANALYZE))[0]}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    print(f"\n\n--- Saving detailed results to {output_path} ---")
    try:
        results_df.to_csv(output_path, index=False)
        print("...Successfully saved the output file.")
    except Exception as e:
        print(f"ERROR: Could not save the output file. Error: {e}")

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()
