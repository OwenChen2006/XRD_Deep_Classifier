
## How to Use

1.  **Organize Your Files:** Place your trained model files and the new data CSV into the correct folders as shown in the structure above.

2.  **Configure the Script:** Open the `inference.py` script in a text editor. Update the paths in the **`Configuration`** section at the top of the file to point to your project directory and the specific data file you want to analyze.

3.  **Run from Terminal:**
    *   Open your terminal or command prompt (e.g., PowerShell, Command Prompt, or an Anaconda Prompt).
    *   Navigate to your main project folder:
        ```bash
        cd C:\Path\To\Your\Main\Project\Folder
        ```
    *   Run the script:
        ```bash
        python inference.py
        ```

## Input File Format

The input CSV file (e.g., `45CM-10c_Recon.csv`) must be in a "wide" format:
*   It should **not have a header row**.
*   Each **row** should represent a single pixel.
*   It should have exactly **91 columns**, where each column is the XRD intensity for a specific q-value.

## Output

The script will produce two outputs:

1.  **Terminal Report:** A summary will be printed to the terminal, showing the total number of pixels analyzed and the breakdown of "Benign" vs. "Cancer" classifications. It will also print a detailed, pixel-by-pixel table of the results.

2.  **CSV File:** A new CSV file will be saved in a folder on your Desktop named `XRD_Inference_Results`. This file will contain the detailed, per-pixel results, including:
    *   `Pixel_Index`: The original row number of the pixel.
    *   `Cancer_Probability`: The model's confidence score (0.0 to 1.0).
    *   `Final_Prediction`: The final "Benign" or "Cancer" label based on the optimal threshold.

Possible Models to Load: 
1. XRD_Model, which requires a TSI column as well.
2. Calculated_TSI_Model, which sums all of the q-values manually.
