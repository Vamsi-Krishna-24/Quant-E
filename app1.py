import pickle
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ---------- Load the NSE Data ----------
nse_path = "data/nse_data001.csv"
if not os.path.exists(nse_path):
    st.error(f"NSE CSV not found at {nse_path}")
    st.stop()

nse = pd.read_csv(nse_path)

# Ensure correct columns exist
required_cols = ["Close", "Open", "High", "Low", "Volume", "target"]
if not all(col in nse.columns for col in required_cols):
    st.error("CSV must contain: Close, Open, High, Low, Volume, target")
    st.stop()

# ---------- Sidebar: User Input ----------
st.sidebar.header("User Input")
user_close = st.sidebar.number_input("Closing Price:")
user_open = st.sidebar.number_input("Opening Price:")
user_high = st.sidebar.number_input("High Price:")
user_low = st.sidebar.number_input("Low Price:")
user_volume = st.sidebar.number_input("Volume:")

user_input = pd.DataFrame([{
    "Close": user_close,
    "Open": user_open,
    "High": user_high,
    "Low": user_low,
    "Volume": user_volume,
    "target": None  # Placeholder
}])

# ---------- Train/Test Split as per your algo ----------
train = nse.iloc[:-1400].copy()
test = nse.iloc[-1400:].copy()

# ---------- Append user input as test ----------
full_test = pd.concat([test, user_input], ignore_index=True)
feature_cols = ["Close", "Open", "High", "Low", "Volume"]

# ---------- Train and Predict using fresh model ----------
model = RandomForestClassifier(n_estimators=300, min_samples_split=150, random_state=1)
model.fit(train[feature_cols], train["target"])

try:
    latest_input = full_test.iloc[-1:][feature_cols]
    prediction = model.predict(latest_input)[0]

    st.subheader("📊 Prediction Result")
    if prediction == 0:
        st.markdown('<button style="background-color:red;color:white;padding:10px 20px;border:none;border-radius:5px;">SELL</button>', unsafe_allow_html=True)
        st.error("Trade Recommendation: SELL")
    else:
        st.markdown('<button style="background-color:green;color:white;padding:10px 20px;border:none;border-radius:5px;">BUY</button>', unsafe_allow_html=True)
        st.success("Trade Recommendation: BUY")

except Exception as e:
    st.error(f"Prediction failed: {e}")

# ---------- Optional: Plot ----------
st.subheader("📉 Historical NSE Stock Prices")
try:
    nse_plot = pd.read_csv(nse_path, parse_dates=["Date"])
    nse_plot.set_index("Date", inplace=True)
    fig, ax = plt.subplots()
    ax.plot(nse_plot.index, nse_plot["Close"], label="Closing Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Historical Closing Prices")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"Could not plot historical data: {e}")
