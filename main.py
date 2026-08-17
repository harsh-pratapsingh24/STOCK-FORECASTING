import streamlit as st
from datetime import date
import pandas as pd
import requests
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

st.set_page_config(
    page_title="Stock Forecasting",
    layout="wide"
)
st.markdown("""
<style>

/* Reduce top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 95%;
}

/* Rounded metric cards */
div[data-testid="stMetric"]{
    background-color:#1f2937;
    border-radius:15px;
    padding:18px;
    border:1px solid #374151;
}

/* Section headings */
.section-header{
    font-size:28px;
    font-weight:700;
    margin-top:30px;
    margin-bottom:15px;
}

/* Buttons */
.stButton>button{
    border-radius:10px;
}

/* Dataframes */
[data-testid="stDataFrame"]{
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# Stock name mapping for better user experience
STOCK_NAMES = {
    "ABCAPITAL.NS": "AB Capital",
    "TATSILV.NS": "Tata Silver ETF",
    "WELSPUNLIV.NS": "Welspun Living Limited",
    "CUPID.NS": "CUPID",
    "NFLX": "Netflix",
    "RESPONIND.NS": "Responsive Industries",
    "GOOG": "Alphabet (Google)",
    "RPOWER.NS": "Reliance Power"
}

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

st.title(" Stock Forecasting App")
st.markdown("---")

# Create display names for dropdown
stock_options = [
    f"{STOCK_NAMES.get(ticker, ticker)} ({ticker})"
    for ticker in STOCK_NAMES.keys()
]

with st.sidebar:
    st.header("Configuration")

    selected_stock_display = st.selectbox(
        "Choose Stock",
        stock_options
    )

    n_years = st.slider(
        "Prediction Years",
        1,
        4,
        1
    )

    chart_type = st.radio(
        "Chart Type",
        ["Line", "Candlestick"]
    )

    st.markdown("---")

    st.markdown("""
    <div style="text-align:center;">

    <h3>My Socials</h3>

    <div style="display:flex; justify-content:center; gap:25px;">

    <a href="https://www.linkedin.com/in/harshpratapsingh333/" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg"
             width="42">
    </a>

    <a href="https://leetcode.com/u/harshps/" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/LeetCode_logo_black.png"
             width="42">
    </a>

    </div>

    <br>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------
    # Contact / Feedback Form
    # -------------------------
    st.subheader("📬 Feedback & Stock Request")

    st.caption("Suggest a stock, report a bug, or share feedback.")

    with st.form("feedback_form", clear_on_submit=True):

        name = st.text_input("Name")

        email = st.text_input("Email (Optional)")

        stock = st.text_input(
            "Stock you'd like added",
            placeholder="Example: TCS.NS"
        )

        feedback = st.text_area(
            "Comments / Suggestions",
            placeholder="Tell me what you'd like to see..."
        )

        submitted = st.form_submit_button("Submit")
        

    if submitted:

        if stock == "" and feedback == "":
            st.warning("Please enter a stock request or some feedback.")

        else:

            payload = {
                "name": name,
                "email": email,
                "stock": stock,
                "feedback": feedback
            }

            APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbyAnsT3YJftUoBcVt8ymI2o1LRDpInlS25FuXXLtPJNjlf5ntCBnoMAsbquNW2llmAs/exec"

            try:
                response = requests.post(
                    APPS_SCRIPT_URL,
                    json=payload,
                    timeout=40
                )

                if response.status_code == 200:
                    st.success("Thank you! Your feedback has been submitted.")

                else:
                    st.error("Submission failed. Please try again.")

            except Exception as e:
                st.error(f"⚠️Error: {e}")

        st.markdown("---")
    st.caption("Created by Harsh Pratap Singh")

# Extract the actual ticker
selected_stocks = selected_stock_display.split("(")[-1].rstrip(")")
selected_name = selected_stock_display.split(" (")[0]

period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY, multi_level_index=False)
    if data is None or data.empty:
        st.error(f"No data found for {ticker}")
        return None
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stocks)

if data is None:
    st.stop()

# FORECASTING (needed early for raw data bounds)
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("RAW DATA")
if data is None:
    st.info("No data to display for the selected ticker.")
else:
    yesterday_row = data.iloc[-2]
    yesterday_row = data.iloc[-2] if len(data) > 1 else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Yesterday's Close", f"₹{yesterday_row['Close']:.2f}")
    with col2:
        st.metric("Yesterday's Open", f"₹{yesterday_row['Open']:.2f}")
    with col3:
        st.metric("Yesterday's High", f"₹{yesterday_row['High']:.2f}")
    with col4:
        st.metric("Yesterday's Low", f"₹{yesterday_row['Low']:.2f}")



    st.dataframe(data.tail(10), use_container_width=True)

    # Forecast bounds for yesterday
    if yesterday_row is not None:
        yesterday_forecast = forecast[forecast['ds'].dt.date == yesterday_row['Date'].date()]
        if not yesterday_forecast.empty:
            yf_row = yesterday_forecast.iloc[0]
            st.markdown("**Yesterday's Forecast Bounds (95% CI)**")
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Predicted", f"₹{yf_row['yhat']:.2f}")
            with bc2:
                st.metric("Upper Bound", f"₹{yf_row['yhat_upper']:.2f}")
            with bc3:
                st.metric("Lower Bound", f"₹{yf_row['yhat_lower']:.2f}")

sentiment_df = fetch_news_sentiment(selected_stocks)
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()
def plot_raw_data(data, sentiment_df):
    fig = go.Figure()

    if chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Open"],
                name="Open",
                line=dict(color="#60a5fa"),
                opacity=0.6
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Close"],
                name="Close",
                line=dict(color="#f59e0b", width=2)
            )
        )

    else:
        fig.add_trace(
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price"
            )
        )

    # News markers
    if not sentiment_df.empty:

        prices_at_news = []

        for d in sentiment_df["Date"]:
            past = data[data["Date"] <= d]

            if not past.empty:
                prices_at_news.append(past["Close"].iloc[-1])
            else:
                prices_at_news.append(data["Close"].iloc[0])

        fig.add_trace(
            go.Scatter(
                x=sentiment_df["Date"],
                y=prices_at_news,
                mode="markers",
                name="News",
                marker=dict(
                    size=10,
                    color=sentiment_df["Score"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Sentiment")
                ),
                text=sentiment_df["Title"],
                hovertemplate="<b>%{text}</b><br>Sentiment: %{marker.color:.2f}<extra></extra>"
            )
        )

    fig.update_layout(
        title=f"{selected_name} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=650,
        hovermode="x unified",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

plot_raw_data(data, sentiment_df)

# News sentiment section with better organization
if not sentiment_df.empty:
    st.markdown('<div class="section-header"><h2>📰 Market News & Sentiment</h2></div>', unsafe_allow_html=True)
    
    # Sentiment summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_sentiment = sentiment_df['Score'].mean()
        sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
        st.metric("Average Sentiment", f"{avg_sentiment:.3f}", delta=sentiment_label)
    with col2:
        positive_count = len(sentiment_df[sentiment_df['Score'] > 0.1])
        st.metric("Positive News", f"{positive_count}/{len(sentiment_df)}")
    with col3:
        negative_count = len(sentiment_df[sentiment_df['Score'] < -0.1])
        st.metric("Negative News", f"{negative_count}/{len(sentiment_df)}")
    
    # Recent news in expandable section
    with st.expander("📰 View Latest News Sentiment", expanded=False):
        sentiment_display = sentiment_df[['Date', 'Title', 'Score']].sort_values('Date', ascending=False).head(10)
        
        # Color code sentiment
        def color_sentiment(val):
            color = "#4af973" if val > 0.1 else "#fd1024" if val < -0.1 else "#0c00ec"
            return f'background-color: {color}'
        
        styled_df = sentiment_display.style.map(color_sentiment, subset=['Score'])
        st.dataframe(styled_df, use_container_width=True)
else:
    st.info("📰 No recent news sentiment data available for this stock")

# Forecast results in organized sections
st.markdown('<div class="section-header"><h2>🔮 Forecast Results</h2></div>', unsafe_allow_html=True)

# Forecast metrics in cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label=f"Forecast ({n_years} Year)", 
        value=f"₹{forecast['yhat'].iloc[-1]:.2f}",
        delta=f"{((forecast['yhat'].iloc[-1] / data['Close'].iloc[-1]) - 1) * 100:+.2f}%"
    )
with col2:
    st.metric(
        label="Upper Bound (95%)", 
        value=f"₹{forecast['yhat_upper'].iloc[-1]:.2f}"
    )
with col3:
    st.metric(
        label="Lower Bound (95%)", 
        value=f"₹{forecast['yhat_lower'].iloc[-1]:.2f}"
    )

# Forecast visualization
st.subheader("📈 Price Forecast Chart")

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=m.history["ds"],
    y=m.history["y"],
    mode="lines",
    name="Actual"
))

fig1.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    mode="lines",
    name="Forecast"
))

fig1.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_upper"],
    mode="lines",
    line=dict(width=0),
    showlegend=False
))

fig1.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    mode="lines",
    fill="tonexty",
    line=dict(width=0),
    name="Confidence Interval"
))

fig1.update_layout(
    title={
        'text': f"{selected_name} Stock Price Forecast ({n_years} Year Projection)",
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode='x unified'
)
st.plotly_chart(fig1, use_container_width=True)

# Forecast components and data in expandable sections
col1, col2 = st.columns(2)

with col1:
    with st.expander("📊 View Forecast Data", expanded=False):
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
        forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
        st.dataframe(forecast_display, use_container_width=True)

with col2:
    with st.expander("🔍 View Trend Components", expanded=True):
        fig2 = m.plot_components(forecast)
        fig2.set_size_inches(12, 8)
        st.pyplot(fig2)

# Model information
with st.expander("ℹ️ About the Forecasting Model", expanded=False):
    st.info("""
    **Prophet Model Details:**
    - Developed by Facebook for time series forecasting
    - Automatically detects trends, seasonality, and holiday effects
    - Robust to missing data and outliers
    - Particularly effective with daily data showing multiple seasonal patterns
    - Confidence intervals widen further into the future as uncertainty increases
    
    **Model Parameters:**
    - Training period: {} to {}
    - Forecast horizon: {} days ({} years)
    - Data points used: {}
    """.format(
        df_train['ds'].min().strftime('%Y-%m-%d'),
        df_train['ds'].max().strftime('%Y-%m-%d'),
        period, n_years, len(df_train)
    ))
