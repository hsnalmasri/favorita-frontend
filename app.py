# app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="ETS Demo", layout="centered")

st.title("ETS Forecast (API Demo)")

# --- sidebar inputs ---
with st.sidebar:
    st.header("Parameters")
    Ntest = st.number_input("Backtest months (Ntest)", min_value=1, value=6, step=1)
    horizon = st.number_input("Future horizon (months)", min_value=1, value=3, step=1)
    CI = st.slider("Confidence level", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
    trend = st.selectbox("Trend", options=["add", "mul", None], index=0)
    seasonal = st.selectbox("Seasonal", options=["add", "mul", None], index=1)
    api_url = st.text_input("API URL", value="http://185.158.107.43/ets/tune")

run = st.button("Run ETS")

if run:
    with st.spinner("Calling API..."):
        payload = {
            "Ntest": Ntest,
            "horizon": horizon,
            "CI": CI,
            "trend": trend,
            "seasonal": seasonal,
        }
        r = requests.post(api_url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

    # rebuild DataFrames
    train = pd.DataFrame(data["train"])
    pred = pd.DataFrame(data["pred"])
    metrics = data["metrics"]

    st.subheader("Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("SMAPE", f'{metrics.get("smape", float("nan")):.3f}')
    col2.metric("MAE", f'{metrics.get("mae", float("nan")):.3f}')
    col3.metric("Bias", f'{metrics.get("bias", float("nan")):.3f}')

    # ensure ds is datetime for plotting
    for df in (train, pred):
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])

    st.subheader("Backtest (Train vs Fitted)")
    if {"ds", "y", "fitted"}.issubset(train.columns):
        plot_df = train[["ds", "y", "fitted"]].set_index("ds").sort_index()
        st.line_chart(plot_df)
    else:
        st.info("train DataFrame missing one of: ds, y, fitted")

    st.subheader("Holdout & Forecast")
    need = {"ds", "y", "yhat", "lower", "upper"}
    if need.issubset(pred.columns):
        plot_df = pred[["ds", "y", "yhat"]].set_index("ds").sort_index()
        st.line_chart(plot_df)
        # show table with CIs for transparency
        with st.expander("Show prediction table"):
            st.dataframe(pred.sort_values("ds").reset_index(drop=True))
    else:
        st.info(f"pred DataFrame missing columns: {sorted(list(need - set(pred.columns)))}")
