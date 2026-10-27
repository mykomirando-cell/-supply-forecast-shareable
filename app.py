import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import warnings
import os
from sklearn.metrics import mean_absolute_error
import io

warnings.filterwarnings("ignore")

# Custom CSS for neon fun theme
st.markdown(
    """
    <style>
    /* Base font size */
    html, body, [class*="css"]  {
        font-size: 20px !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-size: 26px !important;
        color: var(--neon-violet) !important;
    }
    /* Neon colors */
    :root {
        --neon-green: #39ff14;
        --neon-violet: #ff00ff;
        --neon-pink: #ff1493;
        --background: #0a0a0a;
    }
    .stApp {
        background-color: var(--background);
        color: var(--neon-green);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--neon-violet) !important;
    }
    .stButton>button {
        background-color: var(--neon-pink);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff69b4;
    }
    .stSelectbox, .stMultiSelect, .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: var(--neon-green);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper functions for analysis (adapted from inventory_analysis_auto.py)
def analyze_demand(uploaded_file):
    """
    Perform the demand forecasting analysis on the uploaded CSV or Excel file.
    Returns a dictionary with results: dataframes, summary text, plot, etc.
    """
    # Read the uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format")
        st.stop()

    # Determine date and demand columns (handles optional first Item Name column)
    ncols = df.shape[1]
    if ncols >= 3:
        # Assume first column is item name (ignore), second is date, third is demand
        item_name_series = df.iloc[:, 0].dropna()
        item_name = str(item_name_series.iloc[0]) if not item_name_series.empty else "Item"
        date_col = df.columns[1]
        demand_col = df.columns[2]
    elif ncols == 2:
        # No item name column – fallback
        item_name = "Unknown Item"
        date_col = df.columns[0]
        demand_col = df.columns[1]
    else:
        st.error("Unexpected number of columns in the file")
        st.stop()

    # Keep only date and demand columns and rename
    df = df[[date_col, demand_col]].copy()
    df.columns = ["date", "demand"]

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Store raw demand for statistics and outlier detection
    raw_demand = df["demand"].copy()
    mean_raw = raw_demand.mean(skipna=True)
    std_raw = raw_demand.std(skipna=True)
    lower_limit = mean_raw - 3 * std_raw
    upper_limit = mean_raw + 3 * std_raw

    # Forward-fill missing values
    df["demand"] = df["demand"].ffill()
    filled_demand = df["demand"].copy()

    # Remove outliers using raw data mean and std (3-sigma rule)
    df = df[
        (df["demand"] < mean_raw + 3 * std_raw)
        & (df["demand"] > mean_raw - 3 * std_raw)
    ]

    cleaned_demand = df["demand"].copy()

    # Derive date range string for titles
    min_date = df["date"].min()
    max_date = df["date"].max()
    span_days = (max_date - min_date).days

    if span_days < 31:
        date_range_str = f"over the last {span_days} day{'s' if span_days != 1 else ''}"
    elif span_days < 365:
        approx_months = round(span_days / 30.44, 1)
        date_range_str = f"over the last {approx_months} month{'s' if approx_months != 1 else ''}"
    else:
        approx_years = round(span_days / 365.25, 2)
        date_range_str = f"over the last {approx_years} year{'s' if approx_years != 1 else ''}"

    # Prepare features for Linear Regression
    df_lr = df.copy()
    df_lr["day"] = df_lr["date"].dt.day
    df_lr["month"] = df_lr["date"].dt.month
    df_lr["weekday"] = df_lr["date"].dt.weekday  # Monday=0
    df_lr["lag_7"] = df_lr["demand"].shift(7)
    df_lr = df_lr.dropna(subset=["lag_7"]).reset_index(drop=True)

    # Prepare time series for ETS
    ts = df.set_index("date")["demand"]
    ts = ts.groupby(level=0).mean()  # Handle duplicate dates
    ts = ts.asfreq('D').ffill()

    # Train/test split: holdout last 28 days
    holdout_days = 28
    if len(df_lr) < holdout_days + 7:  # need at least 7 days for lag feature
        train_lr = df_lr
        test_lr = pd.DataFrame(columns=df_lr.columns)
        train_ts = ts
        test_ts = pd.Series(dtype=float)
    else:
        train_lr = df_lr.iloc[:-holdout_days]
        test_lr = df_lr.iloc[-holdout_days:]
        train_ts = ts.iloc[:-holdout_days]
        test_ts = ts.iloc[-holdout_days:]

    # Fit Linear Regression on training data
    feature_cols = ["day", "month", "weekday", "lag_7"]
    X_train = train_lr[feature_cols].fillna(0)
    y_train = train_lr["demand"]
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Fit ETS on training data
    try:
        ets_model = ETSModel(
            train_ts,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=7
        )
    except Exception:
        # fall back to non-seasonal if data too short
        ets_model = ETSModel(
            train_ts,
            error="add",
            trend="add",
            seasonal=None
        )
    ets_fit = ets_model.fit()

    # Generate holdout forecasts
    # Linear Regression holdout forecast
    if not test_lr.empty:
        X_test = test_lr[feature_cols].fillna(0)
        lr_forecast_holdout = lr_model.predict(X_test) if len(X_test) > 0 else np.array([])
    else:
        lr_forecast_holdout = np.array([])

    # ETS holdout forecast
    if len(test_ts) > 0:
        start_idx = len(train_ts)
        end_idx = start_idx + len(test_ts) - 1
        pred_obj = ets_fit.get_prediction(start=start_idx, end=end_idx)
        ets_forecast_holdout = pred_obj.predicted_mean.values
        ets_forecast_holdout = np.maximum(ets_forecast_holdout, 0)
    else:
        ets_forecast_holdout = np.array([])

    # Evaluate MAE on holdout (if holdout exists)
    if len(test_lr) > 0 and len(test_ts) > 0:
        mae_lr = mean_absolute_error(test_lr["demand"], lr_forecast_holdout)
        mae_ets = mean_absolute_error(test_ts, ets_forecast_holdout)
    else:
        mae_lr = None
        mae_ets = None

    # Refit selected model on FULL data (we still need both forecasts for the plot)
    # Linear Regression on full data
    X_full = df_lr[feature_cols].fillna(0)
    y_full = df_lr["demand"]
    lr_model_full = LinearRegression()
    lr_model_full.fit(X_full, y_full)

    # Get last valid date from df_lr
    valid_dates_lr = df_lr["date"].dropna()
    if valid_dates_lr.empty:
        last_date_hist = ts.index.max() if not ts.empty else pd.Timestamp('2026-01-01')
    else:
        last_date_hist = valid_dates_lr.iloc[-1]
    future_dates = pd.date_range(start=last_date_hist + pd.Timedelta(days=1), periods=28, freq="D")

    # Iterative lag for LR
    hist_vals = list(df_lr["demand"].iloc[-7:])
    lr_forecast_vals = []
    for fd in future_dates:
        feat = {
            "day": fd.day,
            "month": fd.month,
            "weekday": fd.weekday(),
            "lag_7": hist_vals[-7] if len(hist_vals) >= 7 else hist_vals[-1]
        }
        X_fut = pd.DataFrame([feat])[feature_cols]
        pred = lr_model_full.predict(X_fut)[0]
        pred = max(pred, 0)                     # no negative demand
        lr_forecast_vals.append(pred)
        hist_vals.append(pred)

    lr_forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": lr_forecast_vals
    })

    # ETS on full data
    try:
        ets_model_full = ETSModel(
            ts,
            error="add",
            trend="add",
            seasonal="add",
            seasonal_periods=7
        )
    except Exception:
        ets_model_full = ETSModel(
            ts,
            error="add",
            trend="add",
            seasonal=None
        )
    ets_fit_full = ets_model_full.fit()

    # get_prediction gives us the mean and confidence intervals
    start_idx = len(ts)
    end_idx = len(ts) + 28 - 1
    pred_obj = ets_fit_full.get_prediction(start=start_idx, end=end_idx)
    ets_mean = pred_obj.predicted_mean.clip(lower=0)
    ets_ci80 = pred_obj.pred_int(alpha=0.20).clip(lower=0)
    ets_ci95 = pred_obj.pred_int(alpha=0.05).clip(lower=0)

    # Tidy column names
    ets_ci80 = ets_ci80.rename(columns={
        'lower PI (alpha=0.200000)': 'lower_80',
        'upper PI (alpha=0.200000)': 'upper_80'
    })
    ets_ci95 = ets_ci95.rename(columns={
        'lower PI (alpha=0.050000)': 'lower_95',
        'upper PI (alpha=0.050000)': 'upper_95'
    })

    ets_forecast_df = pd.DataFrame({
        "date": pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=28, freq="D"),
        "forecast": ets_mean.values,
        "lower_80": ets_ci80['lower_80'].values,
        "upper_80": ets_ci80['upper_80'].values,
        "lower_95": ets_ci95['lower_95'].values,
        "upper_95": ets_ci95['upper_95'].values
    })

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Historical data
    plt.plot(ts.index, ts.values, label="Historical", color="black", linewidth=1.5)
    plt.axhline(y=lower_limit, color="gray", linestyle="--", linewidth=1, label=f"Lower limit (3σ) = {lower_limit:.1f}")
    plt.axhline(y=upper_limit, color="gray", linestyle="--", linewidth=1, label=f"Upper limit (3σ) = {upper_limit:.1f}")

    # Linear Regression forecast
    plt.plot(lr_forecast_df["date"], lr_forecast_df["forecast"],
             label="Linear Regression Forecast", color="steelblue", linewidth=2)

    # ETS forecast
    plt.plot(ets_forecast_df["date"], ets_forecast_df["forecast"],
             label="ETS Forecast", color="darkorange", linewidth=2)

    # 4-week average lines
    four_week_avg_lr = lr_forecast_df["forecast"].mean()
    four_week_avg_ets = ets_forecast_df["forecast"].mean()
    forecast_start = lr_forecast_df["date"].iloc[0]
    forecast_end = lr_forecast_df["date"].iloc[-1]
    plt.hlines(y=four_week_avg_lr, xmin=forecast_start, xmax=forecast_end,
               colors="navy", linestyles="--", linewidth=2,
               label=f"LR 4-wk avg = {four_week_avg_lr:,.1f}")
    plt.hlines(y=four_week_avg_ets, xmin=forecast_start, xmax=forecast_end,
               colors="sienna", linestyles=":", linewidth=2,
               label=f"ETS 4-wk avg = {four_week_avg_ets:,.1f}")

    # Cosmetics
    plt.title(f"Demand Forecast for {item_name} - Next 28 Days (Model Comparison) {date_range_str}")
    plt.ylabel("Units")
    plt.xlabel("Date")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Save plot to a bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150)
    img_buffer.seek(0)
    plt.close()

    # Prepare forecast dataframe for download (merge both models)
    forecast_df = lr_forecast_df.copy()
    forecast_df = forecast_df.rename(columns={'forecast': 'forecast_lr'})
    ets_cols = ets_forecast_df[['date', 'forecast', 'lower_80', 'upper_80', 'lower_95', 'upper_95']].rename(
        columns={'forecast': 'forecast_ets'}
    )
    forecast_df = forecast_df.merge(ets_cols, on='date', how='left')
    # Add item name column for reference
    forecast_df.insert(0, "Item", item_name)

    # Prepare cleaned history for download (with features)
    df_out = df_lr.copy()
    df_out.insert(0, "Item", item_name)

    # Generate insights summary markdown
    insights_md = f"""# Insights Summary - Automatic Model Selection (Both Models Included)

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Analyst:** Supply Planning Team
**Dataset:** Uploaded demand data
**Item:** {item_name}

---

## 1 Key Metrics from Historical Data
- **Average weekly demand (last 4 weeks):** {cleaned_demand.rolling(7).mean().iloc[-1]*7:.1f} units
- **Outlier removal limits (mean ± 3σ):** [{mean_raw - 3*std_raw:.1f}, {mean_raw + 3*std_raw:.1f}]

## 2 Forecast Output (Next 28 days)
- **Linear Regression Forecast Method:** Linear Regression
- **Predicted total demand over next 28 days (LR):** {lr_forecast_df['forecast'].sum():.0f} units
- **Average weekly forecast (LR):** {lr_forecast_df['forecast'].sum()/4:.1f} units/week
- **Trend (LR):** linear trend from regression.

- **ETS Forecast Method:** ETS (with 80% and 95% prediction intervals)
- **Predicted total demand over next 28 days (ETS):** {ets_forecast_df['forecast'].sum():.0f} units
- **Average weekly forecast (ETS):** {ets_forecast_df['forecast'].sum()/4:.1f} units/week
- **Trend (ETS):** see plot.
- **ETS 80% PI range for day 28:** [{ets_forecast_df.iloc[-1]['lower_80']:.1f}, {ets_forecast_df.iloc[-1]['upper_80']:.1f}]
- **ETS 95% PI range for day 28:** [{ets_forecast_df.iloc[-1]['lower_95']:.1f}, {ets_forecast_df.iloc[-1]['upper_95']:.1f}]

## 3 Implications for Planning
- If forecasted weekly demand > current average, consider increasing replenishment frequency or raising safety stock.
- If forecast shows a declining trend, opportunities exist to reduce excess inventory and lower carrying costs.

## 4 Recommended Action
> **Action:** Consider both models; Linear Regression showed lower hold-out MAE ({mae_lr if mae_lr is not None else 'N/A':.3f}) while ETS had higher MAE ({mae_ets if mae_ets is not None else 'N/A':.3f}). Use Linear Regression for point forecasts, and refer to ETS for uncertainty intervals if needed.
> **Why:** The script provides both forecasts so you can choose based on your needs for accuracy vs. uncertainty quantification.
> **Next Steps:**
> 1. Use this tool with your actual ERP/WMS export (SKU-level on-hand + sales).
> 2. Rerun this analysis periodically to update forecasts.
> 3. Share the insights with the supply-planning team for feedback.
"""

    # Return results
    return {
        'lr_forecast_df': lr_forecast_df,
        'ets_forecast_df': ets_forecast_df,
        'forecast_df': forecast_df,
        'cleaned_history_df': df_out,
        'insights_md': insights_md,
        'plot_img': img_buffer,
        'mae_lr': mae_lr,
        'mae_ets': mae_ets,
        'date_range_str': date_range_str,
        'mean_raw': mean_raw,
        'std_raw': std_raw,
        'cleaned_demand_mean': cleaned_demand.mean(),
        'item_name': item_name
    }

# Streamlit app layout
st.set_page_config(
    page_title="Supply Chain Forecasting Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Supply Chain Demand Forecasting Tool")
st.markdown("""
Upload your daily demand data (CSV or Excel with date and demand columns) to generate forecasts using
Linear Regression and Exponential Smoothing (ETS) models. The tool provides:
- Side-by-side model comparison
- Forecasts with uncertainty intervals (ETS)
- Key metrics and insights for planning
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

# Download template
with open("sample_data_template.xlsx", "rb") as f:
    template_bytes = f.read()
st.download_button(
    label="📥 Download Template (Excel)",
    data=template_bytes,
    file_name="sample_data_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

if uploaded_file is not None:
    # Show a spinner while processing
    with st.spinner('Analyzing your data...'):
        try:
            results = analyze_demand(uploaded_file)

            # Display success message
            st.success("Analysis completed successfully!")

            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast Plot", "📋 Summary", "💾 Downloads", "📝 Insights"])

            with tab1:
                st.subheader(f"Demand Forecast Comparison for {results['item_name']}")
                # Display the plot
                st.image(results['plot_img'], width='stretch')

                # Show some metrics below the plot
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Last 7-day Avg Demand",
                        value=f"{results['cleaned_demand_mean']:.1f} units/day"
                    )
                with col2:
                    st.metric(
                        label="Implied Weekly Demand",
                        value=f"{results['cleaned_demand_mean']*7:.1f} units/week"
                    )
                with col3:
                    st.metric(
                        label="Outlier Limits (3σ)",
                        value=f"[{results['mean_raw'] - 3*results['std_raw']:.1f}, {results['mean_raw'] + 3*results['std_raw']:.1f}]"
                    )

            with tab2:
                st.subheader("Analysis Summary")
                # Display metrics in a nice format
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Forecast Totals (Next 28 Days)")
                    st.write(f"- **Linear Regression:** {results['lr_forecast_df']['forecast'].sum():.0f} units")
                    st.write(f"- **ETS:** {results['ets_forecast_df']['forecast'].sum():.0f} units")
                with col2:
                    st.markdown("#### Average Weekly Forecast")
                    st.write(f"- **Linear Regression:** {results['lr_forecast_df']['forecast'].sum()/4:.1f} units/week")
                    st.write(f"- **ETS:** {results['ets_forecast_df']['forecast'].sum()/4:.1f} units/week")

                st.markdown("#### Holdout MAE (Model Accuracy)")
                if results['mae_lr'] is not None and results['mae_ets'] is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Linear Regression MAE", f"{results['mae_lr']:.3f}")
                    with col2:
                        st.metric("ETS MAE", f"{results['mae_ets']:.3f}")
                    st.caption("Lower MAE indicates better accuracy on the holdout period.")
                else:
                    st.warning("Insufficient data for holdout MAE calculation.")

                st.markdown("#### ETS Prediction Intervals (Day 28)")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**80% PI:** [{results['ets_forecast_df'].iloc[-1]['lower_80']:.1f}, {results['ets_forecast_df'].iloc[-1]['upper_80']:.1f}]")
                with col2:
                    st.write(f"**95% PI:** [{results['ets_forecast_df'].iloc[-1]['lower_95']:.1f}, {results['ets_forecast_df'].iloc[-1]['upper_95']:.1f}]")

            with tab3:
                st.subheader("Download Results")

                # Create download buttons for each file
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    # Cleaned history CSV
                    csv_history = results['cleaned_history_df'].to_csv(index=False)
                    st.download_button(
                        label="📥 Cleaned History",
                        data=csv_history,
                        file_name="cleaned_history.csv",
                        mime="text/csv"
                    )

                with col2:
                    # Forecast CSV
                    csv_forecast = results['forecast_df'].to_csv(index=False)
                    st.download_button(
                        label="📥 Forecasts",
                        data=csv_forecast,
                        file_name="forecast_28days.csv",
                        mime="text/csv"
                    )

                with col3:
                    # Insights summary MD
                    st.download_button(
                        label="📥 Insights (MD)",
                        data=results['insights_md'],
                        file_name="insights_summary.md",
                        mime="text/markdown"
                    )

                with col4:
                    # Plot image
                    st.download_button(
                        label="📥 Forecast Plot (PNG)",
                        data=results['plot_img'],
                        file_name="demand_forecast.png",
                        mime="image/png"
                    )

                st.info("💡 Tip: Right-click on the download buttons to save files to your computer.")

            with tab4:
                st.subheader("Detailed Insights")
                st.markdown(results['insights_md'])

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.exception(e)

else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload a CSV or Excel file to begin the analysis.")

    # Show expected format
    st.markdown("""
    ### Expected File Format (CSV or Excel)
    Your file should contain at least two or three columns:
    - Optional first column: **Item Name** (will be shown in reports but ignored for modeling)
    - Second column: a date column (named 'Date' or will be treated as the first date column)
    - Third column: a demand/numeric column (named 'Demand', 'Daily Issuance', etc. or will be treated as the second data column)

    Example format (with Item Name):
    ```
    Item Name,Date,Daily Issuance
    Marinated Chicken,2026-08-01,3.331728
    ,2026-08-02,6.192852
    ,2026-08-03,5.098488
    ...
    ```

    Or without Item Name (two columns):
    ```
    Date,Demand
    8/1/2026,3.331728
    8/2/2026,6.192852
    8/3/2026,5.098488
    ...
    ```

    The tool will:
    1. Automatically detect and rename columns
    2. Handle missing values (forward-fill)
    3. Remove outliers using 3-sigma rule based on raw data
    4. Generate forecasts for the next 28 days
    5. Compare Linear Regression and ETS models
    """)

    # Show sample data expander
    with st.expander("📄 See sample data format"):
        st.code("""Item Name,Date,Daily Issuance
Marinated Chicken,2026-08-01,3.331728
,2026-08-02,6.192852
,2026-08-03,5.098488
,2026-08-04,9.677280
,2026-08-05,4.581828
""")

# Footer
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built by Myko with Claude ❤️ using Streamlit • For Supply Planning team</p>
        <p>© 2026 Supply Analytics Demo</p>
    </div>
    """,
    unsafe_allow_html=True
)