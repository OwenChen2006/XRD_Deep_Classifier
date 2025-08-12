# AI-Driven Breast Cancer Detection in X-ray Diffraction Images

**Version:** 1.0.0
**Lead Researcher:** Owen Chen, Calidar Medical
**Date:** July 23, 2025

## 1. Project Overview

This repository contains the complete codebase and trained model artifacts for the research project: "A Hybrid Deep Learning Approach for Breast Cancer Identification in Spatially Resolved X-ray Diffraction Data."

The primary achievement of this work is a state-of-the-art, pixel-level **1D Convolutional Neural Network (1D-CNN)** that achieves an exceptional **ROC AUC of 0.995** for classifying cancerous vs. benign tissue from hyperspectral XRD data. This model was trained and validated on a large dataset of over 900,000 tissue pixels from 22 real human lumpectomy cases.

This document serves as a comprehensive guide to understanding the model, reproducing the results, and applying it in a research or clinical setting.

---

## 2. How the Model Works: A Conceptual Overview

The model is designed to analyze hyperspectral XRD data, where a full 91-point diffraction spectrum is available for every single pixel of a scanned tissue sample. It uses a **multi-input deep learning architecture** to make a highly informed prediction for each pixel.

### The Two-Branch Architecture

The model's "brain" is a 1D-CNN with two parallel branches that analyze different types of data simultaneously:

1.  **The Spectral Expert (1D-CNN Branch):** This is the core of the model. It takes the 91-point XRD spectrum for a pixel as input. Through a series of convolutional layers, it automatically learns to recognize the subtle, characteristic "fingerprints" of different tissue types. It learns to identify patterns in peak shape, location, and width that distinguish cancerous tissue (disordered, dense collagen) from benign tissue (ordered, fatty adipose structures).

2.  **The Spatial Expert (Auxiliary Branch):** This branch takes the **Total Scatter Intensity (TSI)** for the same pixel as input. The TSI is a single number representing the overall tissue density at that location.

These two streams of information are then merged in a final "Decision Head," which weighs the evidence from both the spectral "fingerprint" and the tissue "density" to produce a final, highly accurate probability of cancer.

### Key Training Innovations

*   **Focal Loss:** To overcome the severe class imbalance (cancer pixels are rare), the model was trained with a **Focal Loss** function. This forces the model to focus its learning on the difficult and rare cancer cases, preventing it from getting "lazy" by simply predicting the majority benign class.
*   **Intelligent Callbacks:** The training process was governed by `EarlyStopping` and `ReduceLROnPlateau` to prevent overfitting and ensure the model converged to the most optimal solution, saving the version that performed best on an unseen validation set.

---

## 3. Project Structure

For the scripts to function correctly, your project must be organized in the following folder structure.


**Folder Descriptions:**
*   `1_Raw_Data`: Contains the original `SpatialData.mat` file.
*   `2_Trained_Model_Artifacts`: Contains the final, trained model (`.h5` file) and its associated artifacts (`.joblib` file).
*   `3_Training_Script`: Contains the script used to train the model from scratch.
*   `4_Inference_and_Visualization`: Contains the script for using the trained model to analyze cases and will be the save location for output images.

---

## 4. How to Use the Model (Inference)

This is the primary use case for a research or medical setting: taking the trained model and using it to analyze a lumpectomy case.

### Prerequisites

*   A Python environment with all necessary libraries installed: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `h5py`, `matplotlib`, `seaborn`, `opencv-python`.
*   The project folder structured as described above.

### Running a Single Case Visualization

1.  **Navigate** to the `4_Inference_and_Visualization` directory.
2.  **Open** the `visualize_single_case.py` script in a code editor.
3.  **Configure the `CASE_TO_ANALYZE` variable** at the top of the script to the case you wish to see (e.g., `'Case_5'`).
4.  **Run the script** from your terminal: `python visualize_single_case.py`

### Interpreting the Output

The script will produce and save two key visualizations in the `visualizations` subfolder:

1.  **Visual Maps (`Case_X_DL_Visual_Maps.png`):**
    *   **Left Panel (Mean q-value):** A false-color image representing the average molecular structure.
    *   **Right Panel (Classification Results):** The final classification map. This is the most important output. It shows the spatial location of all True Positives (Cancer found), True Negatives (Benign found), False Positives (Benign mistaken for Cancer), and False Negatives (Cancer missed).

2.  **Spectra Comparison (`Case_X_DL_Spectra_Comparison.png`):**
    *   This 4-panel plot shows the *average XRD spectrum* for each of the four categories (TP, TN, FP, FN). It allows a physicist or researcher to understand *why* the model is making its decisions by visualizing the spectral differences between correctly and incorrectly classified pixels. For example, the "False Positives" plot will show the average spectrum of benign tissue that looks confusingly similar to a cancerous spectrum.

---

## 5. Retraining the Model (Optional)

If you acquire new data and wish to retrain or fine-tune the model, use the script provided in the `3_Training_Script` folder.

1.  Ensure your new data is added to the `SpatialData.mat` file or is in an identical format.
2.  Open the `run_training.py` script.
3.  Update the `FILE_PATH` to point to your new data file.
4.  Execute the script: `python run_training.py`

**Note:** Training is computationally intensive and requires a machine with a compatible GPU for a reasonable training time. The process will overwrite the existing model artifacts in the `2_Trained_Model_Artifacts` folder. It is recommended to back up the original artifacts before retraining.

---

## Single pixel predictor:
1. Takes a specific Case ID and (X, Y) coordinate as input.
2. Loads your final trained pixel-level deep learning model and its artifacts.
3. Extracts the data for that single pixel.
4. Runs the full prediction pipeline on that single datapoint.
5. Saves a clear and detailed report of the inputs and outputs to an Excel file.
The New Inference Pipeline for a Single Pixel

Load Artifacts: The script will load the saved FINAL_CNN_FOCAL_LOSS_MODEL.h5, the fitted spectral_scaler and spatial_scaler, and the optimal_threshold.

Extract Datapoint: It will open the SpatialData.mat file, navigate to the specified Case_ID, and extract the 94-page data vector for the exact (X, Y) coordinate.

Normalize: It will apply the loaded scalers to the 91-point spectrum and the TSI value from the extracted data.

Predict: It will run the model on the single normalized datapoint to get a probability score.

Classify: It will apply the loaded optimal_threshold to the probability to get a final "Benign" or "Cancer" classification.

Export to Excel: It will create a two-sheet Excel file:

Sheet 1 ("Summary_Report"): A single row containing the coordinate, the true label, the model's predicted probability, and the final classification.

Sheet 2 ("Full_Spectrum"): The raw 91-point XRD spectrum for that pixel, so you can plot and analyze it yourself.

Definitive Script to Predict a Single Pixel and Export to Excel

**How to Use:**
*Save this code as a new Python file (e.g., predict_single_pixel.py).
*Place it in the 4_Inference_and_Visualization/ folder of your project.
*Configure the four variables at the top:
*CASE_TO_TEST: The case number you want to analyze (e.g., 5).
*X_COORD_TO_TEST: The X-coordinate of the pixel.
*Y_COORD_TO_TEST: The Y-coordinate of the pixel.
*OUTPUT_EXCEL_PATH: The name for the output Excel file.
*Run from the terminal: python predict_single_pixel.py
