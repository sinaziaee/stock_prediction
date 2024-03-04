# Stock Prediction

The aim of this project is to investigate the performance of Machine Learning and especially Natural Language Processing in predicting the stock market direction. 
The basic principle is **buy low and sell high**, but the complexity is in knowing when to buy and when to sell a stock. 

Four types of analysis exist to forecast the markets:
- Fundamental (Long term investors like Warren Buffet)
- Technical (Day Traders approach)
- Quantitative
- Sentiment (The atmosphere around buying and selling)

They each have their own underlying principles, techniques, tool, and strategies, and understanding each one of them and combining the result of all is more optimal than relying solely on one. 

# Sentiment Analysis

# Technical Analysis

Technical analysis is the use of charts and technical indicators to identify trading signals and price patterns. Various technical strategies will be investigated using the most common leading and lagging trend, momentum, volatility and volume indicators including Moving Averages, Moving Average Convergence Divergence (MACD), Stochastic Oscillator, Relative Strength Index (RSI), Money Flow Index (MFI), Rate of Change (ROC), Bollinger Bands, and On-Balance Volume (OBV).

![Amazon Stock](https://raw.githubusercontent.com/sinaziaee/stock_prediction/master/figs/dashboard.png)

# Time Series Analysis

A time series is basically a series of data points ordered in time and is an important factor in predicting stock market trends. In time series forecasting models, time is the independent variable and the goal is to predict future values based on previously observed values.

Stock prices are often non-stationary and may contain trends or volatility but different transformations can be applied to turn the time series into a stationary process so that it can be modelled.

Recurrent Neural Network (RNN) models such as Simple RNN, Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) will also be explored and various machine learning and deep learning models created, trained, tested and optimised.

![LSTM Prediction of BitCoin](https://raw.githubusercontent.com/sinaziaee/stock_prediction/master/figs/time_series_analysis.png)

# Data Sources

The dataset is obtained by using the apis in yahoo finance and web scraping on investing.com on the news for the 30 trending stocks in New York Stock Exchange.

[Yahoo! Finance](https://ca.finance.yahoo.com/)
[Investing.com](https://www.investing.com/)

# Python libraries

- Numpy
- Pandas
- Matplotlib
- Mplfinance
- Seaborn
- Plotly
- SciPy
- Statsmodels
- Scikit-learn
- Keras
- TensorFlow
- Yfinance
- Beautiful Soup
- Selenium
- NLTK
- TextBlob
- SpaCy
- Gensim
- BERT
- Hugging Face
- PyTorch
- lightweight_charts
- tabulate