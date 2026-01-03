import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Forecasting App")

stocks = ("RPOWER.NS","NFLX","GOLDBEES.BO","AAPL","GOOG")
selected_stocks = st.selectbox("Select dataset for prediction",stocks)

n_years = st.slider("Years of prediction:",1 , 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY, multi_level_index=False)
    if data is None or data.empty:
        st.error(f"No data found for {ticker}")
        return None
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("LOAD DATA...")
data = load_data(selected_stocks)
data_load_state.text("LOADING DATA...DONE")

st.subheader("RAW DATA")
if data is None:
    st.info("No data to display for the selected ticker.")
else:
    st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y =data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y =data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# FORECASTING 
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m= Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("FORECAST DATA")
if data is None:
    st.info("No data to display for the selected ticker.")
else:
    st.write(forecast.tail())

st.write('FORECAST DATA')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('FORECAST COMPONENTS')
fig2 = m.plot_components(forecast)
st.write(fig2)