WATTWATCH STREAMLIT UI — SETUP

1) In your original Colab notebook, run the artifact-saving cell after confirming your final model/config are saved. Run the artifact ZIP/download cell to download electricity_anomaly_artifacts.zip.
2) Extract the ZIP. Copy isolation_forest_model.joblib and model_config.joblib into the artifacts folder in this project.
3) On Windows, open Command Prompt or PowerShell in this folder and run:

   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m streamlit run app.py

4) Open the local URL Streamlit prints, usually http://localhost:8501.

Expected structure:
WattWatch_Streamlit_UI/
  app.py
  requirements.txt
  artifacts/
    isolation_forest_model.joblib
    model_config.joblib

Input files need either a datetime column or Date + Time, plus Global_active_power and all raw sensor columns expected by your model. The original UCI dataset is recommended. Upload several consecutive days so rolling 6-hour and 24-hour features can be calculated.

The app preserves the notebook's hourly resampling, feature engineering, saved feature order, and score calculation (-model.decision_function). It does not retrain the model.

Default threshold is -0.081625, the 2%-target-FPR candidate from your controlled synthetic benchmark comparison. This is a configurable starting point, not a validated real-world false-positive rate. Original data has no verified anomaly labels.


## Dataset

This project uses the UCI Individual Household Electric Power Consumption dataset.

**Download:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Download and extract the dataset separately. Upload the dataset through the WattWatch dashboard to analyse household electricity consumption. The raw dataset is not included in this repository.