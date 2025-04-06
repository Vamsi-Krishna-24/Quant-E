import pickle
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Load the trained model ----------
file_path = '/Users/surisettivamsikrishna/Downloads/Vamsi Pc/CODES/mark1/Q2/random_forest_model_002.pkl'

model = None
if os.path.exists(file_path):
    with open(file_path, 'rb') as model_file:
        model = pickle.load(model_file)
else:
    st.error(f"The model file was not found at: {file_path}")

# ---------- Streamlit UI ----------
st.title("📈 Stock Price Prediction App")
st.markdown("Enter stock data to predict Buy or Sell recommendations.")

# ---------- Sidebar: User Input ----------
st.sidebar.header("User Input")

user_close = st.sidebar.number_input("Closing Price:")
user_open = st.sidebar.number_input("Opening Price:")
user_high = st.sidebar.number_input("High Price:")
user_low = st.sidebar.number_input("Low Price:")
user_volume = st.sidebar.number_input("Volume:")

user_data = pd.DataFrame({
    'Close': [user_close],
    'Open': [user_open],
    'High': [user_high],
    'Low': [user_low],
    'Volume': [user_volume]
})

st.sidebar.subheader("Entered Data")
st.sidebar.write(user_data)

# ---------- Prediction ----------
if model:
    try:
        prediction = model.predict(user_data)

        st.subheader("📊 Prediction Result")

        if prediction[0] == 0:
            sell_button_html = '''
            <button style="background-color: red; color: white; padding: 10px 20px; 
            border: none; border-radius: 5px;">SELL</button>
            '''
            st.markdown(sell_button_html, unsafe_allow_html=True)
            st.error("Trade Recommendation: SELL")

        else:
            buy_button_html = '''
            <button style="background-color: green; color: white; padding: 10px 20px; 
            border: none; border-radius: 5px;">BUY</button>
            '''
            st.markdown(buy_button_html, unsafe_allow_html=True)
            st.success("Trade Recommendation: BUY")
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.warning("Model not loaded, so prediction is skipped.")

# ---------- Visualize Historical Data ----------
st.subheader("📉 Historical NSE Stock Prices")

try:
    nse_path = "/Users/surisettivamsikrishna/Downloads/Vamsi Pc/CODES/mark1/Q1/nse_data001.csv"
    nse = pd.read_csv(nse_path, parse_dates=["Date"])
    nse.set_index("Date", inplace=True)

    fig, ax = plt.subplots()
    ax.plot(nse.index, nse["Close"], label="Closing Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Historical Closing Prices")
    ax.legend()

    st.pyplot(fig)

except Exception as e:
    st.error(f"Could not load or plot historical data: {e}")
