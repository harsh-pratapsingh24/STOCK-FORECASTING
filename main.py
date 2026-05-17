import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    if not text:
        return 0
    return analyzer.polarity_scores(text)['compound']

@st.cache_data
def fetch_news_sentiment(ticker):
    t = yf.Ticker(ticker)
    news = t.news
    sentiment_data = []
    for article in news:
        content = article.get('content', {})
        title = content.get('title', '')
        summary = content.get('summary', '')
        pub_date = content.get('pubDate', '')
        
        score = get_sentiment_score(title + " " + summary)
        sentiment_data.append({
            'Date': pd.to_datetime(pub_date).tz_localize(None),
            'Score': score,
            'Title': title
        })
    return pd.DataFrame(sentiment_data)

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Forecasting App")

stocks = ("NFLX","GOLDBEES.BO","AAPL","GOOG","RPOWER.NS")
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

sentiment_df = fetch_news_sentiment(selected_stocks)

def plot_raw_data(data, sentiment_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y =data['Open'], name='stock_open', line=dict(color='deepskyblue'), opacity=0.5))
    fig.add_trace(go.Scatter(x=data['Date'], y =data['Close'], name='stock_close', line=dict(color='orange')))
    
    if not sentiment_df.empty:
        # Align sentiment dates with stock price dates for overlaying
        # We use the Close price on the day of the news
        prices_at_news = []
        for d in sentiment_df['Date']:
            past_prices = data[data['Date'] <= d]
            if not past_prices.empty:
                prices_at_news.append(past_prices['Close'].iloc[-1])
            else:
                prices_at_news.append(data['Close'].iloc[0])

        fig.add_trace(go.Scatter(
            x=sentiment_df['Date'],
            y=prices_at_news,
            mode='markers',
            name='News Sentiment',
            marker=dict(
                size=12,
                color=sentiment_df['Score'],
                colorscale='RdYlGn',
                showscale=True,
                reversescale=False,
                colorbar=dict(title="Sentiment", x=1.1)
            ),
            text=sentiment_df['Title'],
            hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Score: %{marker.color:.2f}<extra></extra>"
        ))

    fig.layout.update(title_text="Time Series Data with Sentiment Overlay", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data, sentiment_df)

if not sentiment_df.empty:
    st.subheader("RECENT NEWS SENTIMENT")
    st.write(sentiment_df[['Date', 'Title', 'Score']].sort_values('Date', ascending=False).head(10))

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
