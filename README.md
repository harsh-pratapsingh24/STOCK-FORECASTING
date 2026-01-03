
# Stock Price Forecasting using Prophet and Streamlit

## Overview

This project is a **Stock Price Forecasting web application** built using **Facebook Prophet** for time-series forecasting and deployed using **Streamlit**.
The application allows users to select a stock, visualize historical price trends, and generate **future price predictions** for up to four years.

The project demonstrates the complete workflow of a real-world ML system, including data collection, preprocessing, forecasting, visualization, and deployment.

---

## Features

* Fetches real-time historical stock data using Yahoo Finance
* Interactive stock selection
* Time-series visualization of Open and Close prices
* Forecast future stock prices using Prophet
* Forecast horizon selection (1–4 years)
* Forecast trend and seasonality analysis
* Deployed as an interactive Streamlit web application

---

## Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * Streamlit
  * yfinance
  * Prophet
  * Plotly
  * Pandas
  * NumPy
* **Deployment:** Streamlit

---

## Stocks Supported

* RPOWER.NS
* NFLX
* GOLDBEES.BO
* AAPL
* GOOG

---

## Machine Learning Model

* **Model Used:** Prophet (Time-Series Forecasting)
* Prophet models trend, seasonality, and holidays to generate reliable forecasts
* Uses historical closing prices to predict future values
* Automatically handles missing data and trend shifts

---

## Data Pipeline

1. Fetch historical stock data from Yahoo Finance
2. Clean and preprocess data
3. Convert data into Prophet-compatible format (`ds`, `y`)
4. Train Prophet forecasting model
5. Generate future predictions
6. Visualize forecasts and components

---

## Visualizations

* Historical Open vs Close price trends
* Interactive time-series graphs
* Forecasted future stock prices
* Trend and seasonality components from Prophet


## Project Structure

```
├── app.py
├── requirements.txt
├── README.md
```

---

## Future Enhancements

* Add more stocks and indices
* Include technical indicators (RSI, MACD, moving averages)
* Compare Prophet with LSTM and ARIMA models
* Add confidence interval analysis
* Enable real-time forecasting updates

---

## Disclaimer

This project is created for **educational and learning purposes only**.
It should not be considered as financial or investment advice.

---

## Author

**Harsh Pratap Singh**
B.Tech Computer Science Student
Interests: Machine Learning, Data Science, Web Deployment

## CONTACT :
Email : harsh.psingh2005@gmail.com
Linkedin : https://www.linkedin.com/in/harshpratapsingh333/

