# Supply Chain Forecasting Tool (Shareable Version)

This is an easy-to-use, shareable version of the inventory forecasting tool with a web‑based interface. Team members can upload their demand data (CSV or Excel) and get instant forecasts—no manual script running required if deployed online.

## Features
- 📤 Simple CSV **or** Excel file upload (drag‑and‑drop or browse)  
- 📊 Interactive forecast plots comparing Linear Regression and ETS models  
- 📈 Forecasts with 80% and 95% prediction intervals (ETS)  
- 📋 Key metrics and insights summary  
- 💾 One‑click downloads for all results:
  - Cleaned historical data with features  
  - Forecast CSV (both models + intervals)  
  - Insights summary (Markdown)  
  - Forecast plot (PNG)  
- 🎨 Fun neon theme with larger fonts for readability  
- 🏷️ Item name handling (first column shown in reports/plots but ignored for modeling)  
- 📥 Downloadable template to avoid format mistakes  

## How to Use

### Option 1: Run Locally (Recommended for Teams with Python)
1. **Prerequisites**: Python 3.8+ installed.  
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   streamlit run app.py
   ```
4. **Open your browser** at `http://localhost:8501`  
5. Upload your CSV or Excel file and get instant results!

### Option 2: Double‑Click Batch File (Auto‑creates venv if needed)
Simply double‑click `run_streamlit.bat`.  
The batch file will:
- Try to use an existing hermes‑agent virtual environment (if present)  
- Otherwise create a temporary `venv` folder, install requirements, and launch the app  
- Open the app in your default browser  

### Option 3: Run Online – No Python Required (Best for teammates without Python)
Deploy the app for free using **Streamlit Community Cloud**; teammates only need a web browser.

#### To deploy on Streamlit Community Cloud:
1. Push this folder to a GitHub repository  
2. Go to [https://share.streamlit.io/](https://share.streamlit.io/)  
3. Sign in with GitHub  
4. Click "New app" and select your repository, branch, and `app.py`  
5. Click "Deploy!" and share the link with your team  
6. Teammates open the link, upload their file, and get results – **no Python installation needed locally**.

## Expected Data Format
Your file should contain **at least two or three columns**:
- **Optional first column**: **Item Name** (will be shown in reports/plots but ignored for modeling)  
- **Second column**: a date column (named 'Date' or will be treated as the first date column)  
- **Third column**: a demand/numeric column (named 'Demand', 'Daily Issuance', etc. or will be treated as the second data column)

**Example (with Item Name):**
```csv
Item Name,Date,Daily Issuance
Marinated Chicken,2026-08-01,3.331728
,2026-08-02,6.192852
,2026-08-03,5.098488
...
```

**Example (without Item Name – two columns):**
```csv
Date,Demand
8/1/2026,3.331728
8/2/2026,6.192852
8/3/2026,5.098488
...
```

The tool will:
1. Automatically detect and rename columns  
2. Handle missing values (forward‑fill)  
3. Remove outliers using 3‑sigma rule based on raw data  
4. Generate forecasts for the next 28 days  
5. Compare Linear Regression and ETS models  

## Output Files
After analysis, you can download:
- `cleaned_history.csv`: Historical data with engineered features (day, month, weekday, lag_7) **plus Item column**  
- `forecast_28days.csv`: Side‑by‑side forecasts from both models with prediction intervals **plus Item column**  
- `insights_summary.md`: Formatted insights report ready to share  
- `demand_forecast.png`: Publication‑quality forecast plot  

## Notes
- The tool assumes daily data. For other frequencies, please contact the developer.  
- For best results, provide at least 2‑3 months of historical data.  
- All processing happens locally in your browser (when deployed via Streamlit Cloud) or on your machine (when run locally) – your data never leaves your control unless you choose to share the downloaded files.  

## Support
For questions or issues, please contact your local analytics champion or the supply planning team lead.

---
*Built with Streamlit • © 2026 Supply Analytics Demo*