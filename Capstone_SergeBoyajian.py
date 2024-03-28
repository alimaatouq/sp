import warnings
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import hydralit_components as hc
from PIL import Image
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing.sequence import TimeseriesGenerator
from datetime import date
from datetime import timedelta
import snscrape.modules.twitter as sntwitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime as dt
from sklearn.metrics import mean_squared_error
import yfinance as yf
from pandas.tseries.offsets import BDay





#########################################Initiate the new Strealit layout- Customized
st.set_page_config(layout="wide")

menu_data = [
    {'label': "Home Page", 'icon': 'bi bi-house'},
    {'label': "Dashboard", "icon": 'bi bi-clipboard-data'},
    {'label': 'Backtesting Sentiment Analysis', 'icon': 'bi bi-twitter'},
    {'label': 'Backtesting Forecasts', 'icon': 'bi bi-activity'},
#     {'label': 'News & Sentiments', 'icon': 'bi bi-caret-right'},
    {'label': 'Tweets & Sentiments', 'icon': 'bi bi-caret-right'},
    {'label': 'Price Forecast', 'icon': 'bi bi-caret-right'}]
over_theme = {'txc_inactive': 'white','menu_background':'rgb(183,142,108)', 'option_active':'white'}

menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    # hide_streamlit_markers=True,
    #sticky_nav=True, #at the top or not
    #sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
)

######################################### HOME PAGE #########################################
padding=50
if menu_id=="Home Page":
    st.markdown(f""" <style>
    .reportview-container .main .block-container{{
    padding-top: {padding}rem;
    padding-right: {padding}rem;
    padding-left: {padding}rem;
    padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
    image, title= st.columns((1,10))
    #with image:
    #st.image("https://i0.wp.com/gbsn.org/wp-content/uploads/2020/07/AUB-logo.png?ssl=1", width=100)
    with title:
        st.markdown("<h1 style='text-align: center; color: black;'>An Unconventional Technical Analysis Tool for the S&P 500", unsafe_allow_html=True)

    col1,col2= st.columns((5,3))
    with col1:
        st.markdown("<h2 style='text-align: center; color: black;'> Introduction", unsafe_allow_html=True)
        st.write("This app provides unconventional means to help understand the sentiment of investors trading the S&P 500. The rationale behind this app is that the stock market is filled with emotional individuals who drive prices of equities. Another valid point is that social media and especially Twitter has become one of the main channels to share news, sentiments, emotions and opinions related to an individual's investing journey. In addition, understanding what drives prices and volatility of the stock market has been an intriguing project for researchers and investors ever since the inception of the investment concept. Machine Learning has a come a long way and has reached an acceptable level of precision when it comes to Natural Langugage Processing (NLP) for sentiment analysis and time series forecasts for the S&P 500 index. However, it is worth noting that **nothing can help investors time the market perfectly**.")

        st.markdown("<h2 style='text-align: center; color: black;'> Methodology", unsafe_allow_html=True)
        st.write("This tool is divided into two main parts: Sentiment Analysis of tweets scraped directly from Twitter with the hashtag 'S&P 500' and a Machine Learning model, Long Sort Term Memory (LSTM) that predicts the S&P 500 index daily closing price. The main goal of this app is to correlate the sentiment of tweets with historical market movements of the S&P500 index to base future expectations of market movements on future tweets and have a robust model forecasting the direction of movement (value wise) of the S&P 500 index. For the sentiment analysis section, more than 1.2 million tweets about the S&P 500 were scraped to cover the period from January 1st 2014 until 14 July 2022 which will help in the backtesting phase. VADER was used as sentiment intensity analyzer on tweets since it is one of the best performing lexicons and analyzers available today. Regarding the forecasts, a tuned LSTM model was developed with a clear and acceptable range of error.")


        st.markdown("<h2 style='text-align: center; color: black;'> Findings", unsafe_allow_html=True)
        st.write("All tweets related to the S&P 500 were scraped from 1 January 2014 until 14 July 2022 for backtesting. The tweets were divided into two datasets: '2014-2018 Full' and '2019-2022 Full' attached with this app. The tweets were divided using the specific dates to capture the impact of force majeure events like the Covid 19 pandemic and the Russia-Ukraine war on the overall correlation between the S&P 500 price movements and the sentiments scores of tweets about the same index. Results showed that from 2014 to 2018, no major correlation was identified except for heavy/high sentiments scores. However, the results have changed drastically on the second part of the scraped tweets: The correlation between the S&P 500 price movements is reflected in the intensity of daily average sentiments of tweets as shown under the tab 'Backtesting Sentiment Analysis'. Finally, to forecast the price movement of the S&P 500, a LSTM model was compiled which returns an RMSE of USD19 on testing data and around USD46 on unseen data after tuning. The model does not forecast perfectly the price movements but provides a great idea on where the S&P 500 might be headed. The main purpose of the LSTM forecast was not to try to get the most accurate results but rather build an acceptable silhouette for the index's price on a 5 days interval.")

    with col2:
        # image = Image.open('C:\\Users\\admin\\Desktop\\AUB\\Capstone\\HD Pics\\redd-3iLBNFje3oM-unsplash.jpg')
        # st.image(image, use_column_width=True)
        st.image("https://raw.githubusercontent.com/SergeBoyajian/Capstonespx/main/redd-3iLBNFje3oM-unsplash.jpg", use_column_width=True)
############################################################# CUSTOMIZED DASHBOARD WHICH INCLUDES THINGS WE USE ON A DAILY BASIS
if menu_id=="Dashboard":
    col1, space, col2= st.columns((1.5,0.25,3))
######################################################################## GLOBAL OVERVIEW
    with col1:
        st.header("Global Overview")
        html_temp = """
        <!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com" rel="noopener" target="_blank"><span class="blue-text">Economy</span></a> by TradingView</div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
  {
  "colorTheme": "dark",
  "dateRange": "12M",
  "showChart": false,
  "locale": "en",
  "largeChartUrl": "",
  "isTransparent": false,
  "showSymbolLogo": true,
  "showFloatingTooltip": false,
  "width": "400",
  "height": "430",
  "tabs": [
    {
      "title": "Indices",
      "symbols": [
        {
          "s": "FOREXCOM:SPXUSD",
          "d": "S&P 500"
        },
        {
          "s": "FOREXCOM:DJI",
          "d": "Dow 30"
        },
        {
          "s": "INDEX:NKY",
          "d": "Nikkei 225"
        },
        {
          "s": "INDEX:DEU40",
          "d": "DAX Index"
        },
        {
          "s": "GLOBALPRIME:UK100",
          "d": "FTSE"
        },
        {
          "s": "SSE:CSI300-HKG"
        },
        {
          "s": "SKILLING:NASDAQ"
        },
        {
          "s": "XETR:0JHG"
        },
        {
          "s": "HSI:HSI.PI.USD"
        },
        {
          "s": "PEPPERSTONE:VIX"
        },
        {
          "s": "ECONOMICS:EUUR"
        }
      ],
      "originalTitle": "Indices"
    },
    {
      "title": "Futures",
      "symbols": [
        {
          "s": "CME_MINI:ES1!",
          "d": "S&P 500"
        },
        {
          "s": "COMEX:GC1!",
          "d": "Gold"
        },
        {
          "s": "NYMEX:CL1!",
          "d": "Crude Oil"
        },
        {
          "s": "NYMEX:NG1!",
          "d": "Natural Gas"
        }
      ],
      "originalTitle": "Futures"
    },
    {
      "title": "Economy",
      "symbols": [
        {
          "s": "ECONOMICS:USIRYY"
        },
        {
          "s": "ECONOMICS:USCCPI"
        },
        {
          "s": "FRED:CPIAUCSL"
        },
        {
          "s": "FRED:FEDFUNDS"
        },
        {
          "s": "FRED:UNRATE"
        },
        {
          "s": "ECONOMICS:GBIRYY"
        },
        {
          "s": "ECONOMICS:USCCI"
        },
        {
          "s": "ECONOMICS:EUIRYY"
        }
      ]
    }
  ]
}
  </script>
</div>
<!-- TradingView Widget END -->"""

        st.components.v1.html(html_temp, height=400, scrolling=False)

    with space:
        st.header(" ")


######################################################################################## WATCHLIST
    with col2:

        st.header("Watchlist")
        html_temp1="""
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div id="tradingview_0e19c"></div>
  <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/AAPL/" rel="noopener" target="_blank"><span class="blue-text">Apple</span></a> by TradingView</div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.MediumWidget(
  {
  "symbols": [
    [
      "Apple",
      "AAPL|1D"
    ],
    [
      "Google",
      "GOOGL|1D"
    ],
    [
      "Microsoft",
      "MSFT|1D"
    ],
    [
      "XETR:1COV|12M"
    ],
    [
      "NASDAQ:NFLX|12M"
    ],
    [
      "NYSE:VFC|12M"
    ],
    [
      "NYSE:VALE|12M"
    ],
    [
      "LSE:ABF|12M"
    ]
  ],
  "chartOnly": false,
  "width": 1100,
  "height": "430",
  "locale": "en",
  "colorTheme": "dark",
  "isTransparent": false,
  "autosize": false,
  "showVolume": false,
  "hideDateRanges": false,
  "scalePosition": "right",
  "scaleMode": "Normal",
  "fontFamily": "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
  "noTimeScale": false,
  "valuesTracking": "1",
  "chartType": "line",
  "fontColor": "#787b86",
  "gridLineColor": "rgba(240, 243, 250, 0.06)",
  "lineWidth": 3,
  "container_id": "tradingview_0e19c"
}
  );
  </script>
</div>
<!-- TradingView Widget END -->"""
        st.components.v1.html(html_temp1, height=400, scrolling=False)




    col3, col4, col5= st.columns(3)
################################################## EXCHANGE RATES
    with col3:
        st.header("ForEx")
        html_temp1="""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/markets/currencies/forex-cross-rates/" rel="noopener" target="_blank"><span class="blue-text">Exchange Rates</span></a> by TradingView</div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-forex-cross-rates.js" async>
          {
          "width": "400",
          "height": "430",
          "currencies": [
            "EUR",
            "USD",
            "GBP",
            "CHF"
          ],
          "isTransparent": false,
          "colorTheme": "dark",
          "locale": "en"
        }
          </script>
        </div>
        <!-- TradingView Widget END -->"""
        st.components.v1.html(html_temp1, height=400, scrolling=False)
    with col4:
        ################################################ ECON CALENDAR
        st.header("Economic Calendar")
        html_temp1="""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/markets/currencies/economic-calendar/" rel="noopener" target="_blank"><span class="blue-text">Economic Calendar</span></a> by TradingView</div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
        {
        "colorTheme": "dark",
        "isTransparent": false,
        "width": "400",
        "height": "430",
        "locale": "en",
        "importanceFilter": "-1,0,1",
        "currencyFilter": "USD,EUR,DEM,FRF,CNY,GBP"
        }
        </script>
        </div>
        <!-- TradingView Widget END -->"""
        st.components.v1.html(html_temp1, height=400, scrolling=False)

    with col5:
        ########################################################News Widget
        st.header("Latest News")
        html_temp1="""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/key-events/" rel="noopener" target="_blank"><span class="blue-text">Daily news roundup</span></a> by TradingView</div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
        {
        "feedMode": "all_symbols",
        "colorTheme": "dark",
        "isTransparent": false,
        "displayMode": "regular",
        "width": "400",
        "height": "430",
        "locale": "en"
        }
        </script>
        </div>
        <!-- TradingView Widget END -->"""
        st.components.v1.html(html_temp1, height=400, scrolling=False)

######################################### Backtesting Tweets #########################################
if menu_id=="Backtesting Sentiment Analysis":

    tab1, tab2, tab3 = st.tabs(["Overview", "2014-2018", "2019-2022"])
# Sentiment Analysis vs S&P general overview
    with tab1:
        st.header("Overview")
        st.header("Upload Your File (either '2014-2018 Full' OR '2019-2022 Full')")
        uploaded_file=st.file_uploader(label="Upload your Data", accept_multiple_files=False)
        col1, col2= st.columns(2)
        with col1:
            if uploaded_file is not None:
                data=pd.read_csv(uploaded_file)
                data["Dates"]=pd.to_datetime(data["Dates"])
                st.header("Tweets Sentiment")
                start=st.date_input(
                "Start Date",
                datetime.date(2020, 1, 1))
                end=st.date_input(
                "End Date",
                datetime.date(2022, 7, 14))
                mask = (data['Dates'] > str(start)) & (data['Dates'] <= str(end))
                data=data.loc[mask]

                color_discrete_map = {'Neutral': 'rgb(219,199,182)', 'Positive': 'rgb(219,199,182)', 'Negative':'rgb(219,199,182)'}
                d = data.groupby(by=["Sentiment"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
                fig1=px.bar(d, x="Counts", y="Sentiment", color="Sentiment", color_discrete_map= color_discrete_map)
                fig1.update_traces(textfont_size=14, textangle=90, textposition="inside", cliponaxis=False)
                fig1.update_layout(template='simple_white',showlegend=False, autosize=False,width=650, height=400,margin=dict(
                l=0,
                r=0,
                b=0,
                t=10,
                pad=0
                ))
                st.write(fig1, use_container_width=True)



                with col2:
                    st.header("S&P 500 Dynamic Chart")
                    # def main():

                    html_temp = """
                    <!-- TradingView Widget BEGIN -->
                    <div class="tradingview-widget-container">
                    <div id="tradingview_e067c"></div>
                    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/FXOPEN-SPXM/" rel="noopener" target="_blank"><span class="blue-text">SPXM Chart</span></a> by TradingView</div>
                    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                    <script type="text/javascript">
                    new TradingView.widget(
                    {
                    "width": 780,
                    "height": 600,
                    "symbol": "FXOPEN:SPXM",
                    "interval": "D",
                    "timezone": "Africa/Cairo",
                    "theme": "dark",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#f1f3f6",
                    "enable_publishing": false,
                    "hide_side_toolbar": false,
                    "allow_symbol_change": true,
                    "container_id": "tradingview_e067c"
                    }
                    );
                    </script>
                    </div>
                    <!-- TradingView Widget END -->"""

                    st.components.v1.html(html_temp, height=610, scrolling=False)
                        # if __name__ == "__main__":
                        #     main()

# Sentiment Analysis vs S&P from 2014 till 2018
    with tab2:
        st.header("Backtesting 2014-2018")
        uploaded_files=st.file_uploader(label="Upload the file named '2014-2018 Full' ", accept_multiple_files=False, type=['csv', 'xlsx'])
        if uploaded_files is not None:
            df_cd=pd.read_csv(uploaded_files)
            df_cd["Dates"]= pd.to_datetime(df_cd["Dates"])
            scaler=MinMaxScaler()
            df_cd["Closing_Price"]= scaler.fit_transform(df_cd[["Close"]])
            # #Groupby date and get average of compound to get average daily sentiment of tweets to be able to compare with closing price of S&P on daily timeframe
            a= df_cd.groupby(by= df_cd["Dates"], as_index=False)["compound"].agg("mean")

            #Plot S&P
            fig = go.Figure()
            fig.add_scatter( x=df_cd['Dates'], y=df_cd['Closing_Price'],
                    mode='lines',
                    name='S&P 500')
            #
            # # Add compound score of sentiment
            fig.add_scatter(x=a['Dates'], y=a['compound'], mode='lines', name="Sentiment Score")
            fig.update_layout( autosize=False,width=1600, height=800, title="Comparison S&P 500 & Tweets Sentiment (2014-2018)" )
            # # Show plot
            st.write(fig)
            corr= df_cd["Closing_Price"].corr(a["compound"])
            st.markdown(f"The correlation between the S&P 500 closing price and the average daily sentiment score of tweets is {corr*100:.2f}%")

# Sentiment Analysis vs S&P from 2019 till 2022
    with tab3:
        st.header("Backtesting 2019-2022")
        uploaded_filess=st.file_uploader(label="Upload the file named '2019-2022 Full' ", accept_multiple_files=False, type=['csv', 'xlsx'])
        if uploaded_filess is not None:
            df_cdd=pd.read_csv(uploaded_filess)
            df_cdd["Dates"]= pd.to_datetime(df_cdd["Dates"])
            scaler=MinMaxScaler()
            df_cdd["Closing_Price"]= scaler.fit_transform(df_cdd[["Close"]])
            # #Groupby date and get average of compound to get average daily sentiment of tweets to be able to compare with closing price of S&P on daily timeframe
            a= df_cdd.groupby(by= df_cdd["Dates"], as_index=False)["compound"].agg("mean")

            #Plot S&P
            fig1 = go.Figure()
            fig1.add_scatter(x=df_cdd['Dates'], y=df_cdd['Closing_Price'],
                    mode='lines',
                    name='S&P 500')
            #
            # # Add compound score of sentiment
            fig1.add_scatter(x=a['Dates'], y=a['compound'], mode='lines', name="Sentiment Score")
            fig1.update_layout(autosize=False,width=1600, height=800, title="Comparison S&P 500 & Tweets Sentiment (2019-2012)")
            # # Show plot
            st.write(fig1)
            correlation= df_cdd["Closing_Price"].corr(a["compound"])
            st.markdown(f"The correlation between the S&P 500 closing price and the average daily sentiment score of tweets is {correlation*100:.2f}% ")



######################################### Backtesting Forecasts #########################################

if menu_id=="Backtesting Forecasts":
    col1,col2=st.columns(2)
    spx=yf.Ticker("^GSPC")
    df = spx.history(start="2002-09-19", end="2022-7-31", interval="1d")
    df.reset_index(inplace=True)
    df1=df.iloc[0:4000,]
    df2=df.iloc[4000:,]

    with col1:
        close_prices = df1.iloc[0:3200,4]
        values = close_prices.values
        training_data_len = math.ceil(len(values))
        scaler = MinMaxScaler(feature_range=(0,1))
        #scaler= StandardScaler()
        scaled_data = scaler.fit_transform(values.reshape(-1,1))


        x_train = []
        y_train = []

        for i in range(5, len(scaled_data)):
            x_train.append(scaled_data[i-5:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


        close_prices_test = df1.iloc[3200:,4]
        values_test = close_prices_test.values
        testing_data_len = math.ceil(len(values_test))
        #scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data_test = scaler.fit_transform(values_test.reshape(-1,1))



        date_values=df1["Date"].iloc[3200:, ].values.reshape(-1,1)
        x_test = []
        y_test=[]
        date_test=[]
        for i in range(5, len(scaled_data_test)):
          x_test.append(scaled_data_test[i-5:i, 0])
          y_test.append(scaled_data_test[i, 0])
          date_test.append(date_values[i,0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        tf.random.set_seed(1)
        model = keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=False, input_shape=(x_train.shape[1], 1)))
        # model.add(layers.LSTM(100, return_sequences=False))
        # model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')
        with st.spinner("LSTM Model is Fitting the Data..."):
            model.fit(x_train, y_train, batch_size= 5, epochs=20)
        st.success("Done!")




        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        y_test=scaler.inverse_transform(y_test.reshape(-1,1))
        rmse= mean_squared_error(y_test, predictions, squared=False)
        print(f" The new RMSE is: {rmse} ")



        date_test = [int(i) for i in date_test]
        y_test=  [int(i) for i in y_test]
        predictions=  [int(i) for i in predictions]

        dataset = pd.DataFrame({'Date': date_test, 'Forecast': predictions, "Actual": y_test}, columns=['Date', 'Forecast', 'Actual'])
        dataset["Date"]= pd.to_datetime(dataset["Date"])


        fig= go.Figure()
        fig.add_scatter(x=dataset["Date"] ,y=dataset['Actual'], name="Actual")

        # # Add compound score of sentiment
        fig.add_scatter(x=dataset["Date"], y=dataset['Forecast'], mode='lines', name= "Forecast")
        fig.update_layout( autosize=False,width=650, height=550, title="Performance on Testing Data" )
        # Show plot
        st.write(fig)
        st.write(f"The training data includes all closing prices of the S&P 500 index from 19 September 2002 until 8 June 2015. Basically, the tuned LSTM model was trained on the mentioned data and told to forecast the closing prices of the index from 8 June 2015 until 8 August 2018. As seen above, the tuned LSTM model was able to forecast the test data of the S&P 500 index closing prices with an RMSE of USD {rmse:.2f} which is much lower than the average closing prices divided by 2 (USD 958.16)")


    with col2:

        values_2 = df2["Close"].values
        val_data_len = math.ceil(len(values_2))
        date_values=df2["Date"].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0,1))
        #scaler= StandardScaler()
        scaled_data_2 = scaler.fit_transform(values_2.reshape(-1,1))

        date_val_2=[]
        x_val_2 = []
        y_val_2 = []

        for i in range(5, len(scaled_data_2)):
            x_val_2.append(scaled_data_2[i-5:i, 0])
            y_val_2.append(scaled_data_2[i, 0])
            date_val_2.append(date_values[i,0])

        x_val_2, y_val_2, date_val_2= np.array(x_val_2), np.array(y_val_2),np.array(date_val_2)

        x_val_2 = np.reshape(x_val_2, (x_val_2.shape[0], x_val_2.shape[1], 1))



        predictions_2 = model.predict(x_val_2)
        predictions_2=scaler.inverse_transform(predictions_2)
        y_val_2=scaler.inverse_transform(y_val_2.reshape(-1,1))
        rmse= mean_squared_error(y_val_2, predictions_2, squared=False)
        print(f"The new RMSE is: {rmse}")



        date_val_2 = [int(i) for i in date_val_2]
        y_val_2=  [int(i) for i in y_val_2]
        predictions_2=  [int(i) for i in predictions_2]

        dataset = pd.DataFrame({'Date': date_val_2, 'Forecast': predictions_2, "Actual": y_val_2}, columns=['Date', 'Forecast', 'Actual'])
        dataset["Date"]= pd.to_datetime(dataset["Date"])


        fig1= go.Figure()
        fig1.add_scatter(x=dataset["Date"] ,y=dataset['Actual'], name="Actual")

        # # Add compound score of sentiment
        fig1.add_scatter(x=dataset["Date"], y=dataset['Forecast'], mode='lines', name= "Forecast")
        fig1.update_layout( autosize=False,width=650, height=550, title="Performance on Unseen Data" )
        # Show plot
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
      
        st.write(fig1)
        st.write(f"The tuned and trained LSTM model was used to directly predict the unseen data passed which includes the index closing prices from 8 August 2018 until end of July 2022. The plot shows the performance of the tuned model in forecasting completely unseen data. As seen above, the tuned LSTM model was able to forecast the validation (unseen data) of the S&P 500 index closing prices with an RMSE of USD {rmse:.2f} which is much lower than the average closing prices divided by 2 (USD 958.16)")


######################################### Predictions: News Headlines, Tweets, S&P 500 Price Chart #########################################

# if menu_id=="News & Sentiments":
#     # pd.set_option('display.max_rows', None)
#     # pd.set_option('display.max_columns', None)
#     # pd.set_option('display.width', None)
#     # pd.set_option('display.max_colwidth', None)
# ######################################### Sentiment Analysis on News Headdlines
#     col, img= st.columns(2)
#     with col:
#         warnings.filterwarnings("ignore", category=DeprecationWarning)

#         options = webdriver.ChromeOptions()
#         options.add_argument("headless")
#         driver = webdriver.Chrome(ChromeDriverManager().install())
#         # driver.implicitly_wait(15)
#         # driver = webdriver.Chrome("./chromedriver", chrome_options=options)

#         st.write("Opening website...")
#         driver.get("https://www.tradingview.com/symbols/SPX/news/")
#         news_container = driver.find_element(By.CLASS_NAME, "grid-gjUO4ZsZ")

#         print("Waiting for news to show...")
#         # WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.TAG_NAME, "a")))
#         WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".grid-gjUO4ZsZ a")))
#         # news_elements = news_container.find_elements(By.TAG_NAME, "a")
#         news_elements = driver.find_elements(By.CSS_SELECTOR, ".grid-gjUO4ZsZ a")
#         # print(news_elements)
#         max_index = len(news_elements)
#         index = 0

#         result = []

#         # st.write("Getting news...")
#         with st.spinner("Scraping All Recent S&P 500 News..."):
#             while index < max_index:
#                 try:
#                     WebDriverWait(driver, 30).until(
#                         EC.element_to_be_clickable((By.CSS_SELECTOR, ".card-BohupzCl.cardLink-BohupzCl")))
#                     news_array = news_container.find_elements(By.CSS_SELECTOR, ".card-BohupzCl.cardLink-BohupzCl")
#                     news_element = news_array[index]
#                     TIME = news_element.find_element(By.TAG_NAME, "relative-time").get_attribute("event-time")
#                     HEADER = news_element.find_element(By.CLASS_NAME, "title-O1eazALv").text
#                     result.append([TIME, HEADER])
#                     index += 1
#                     # if index % 50 == 0:
#                     #     st.write(f"{index / 2}% done")
#                 except:
#                     break
#                     # df = pd.DataFrame(result, columns=["Date", "Header"])
#                     # # dataframe.to_csv("dataframe.csv", index=False)
#                     # st.write(df)
#                     # # exit(f"Code stopped because of an 'element not found' error at {index / 2}%")
#             st.success("Done!")
#             df = pd.DataFrame(result, columns=["Date", "Header"])

#             # dataframe.to_csv("dataframe.csv", index=False)
#             st.write(df)
#             driver.close()
#             driver.quit()
#     #
#         # if st.button("Get Sentiment Analysis"):
#         analyser = SentimentIntensityAnalyzer()
#         # function to calculate polarity scores
#         pol = lambda x: analyser.polarity_scores(x)
#         # creating new column 'polarity' in clean_df
#         df['polarity'] = df['Header'].apply(pol)
#         df1=df['polarity'].apply(pd.Series)
#         df2 = pd.concat([df, df['polarity'].apply(pd.Series)], axis=1)
#         df2['Dates'] = pd.to_datetime(df2['Date']).dt.date
#         df2= df2[df2['Dates'].between(date.today() - timedelta(days=5), date.today())]


#         #Classify tweets based on their compound scores to three labels: Positive, Negative and Neutral
#         def getSentiment(score):
#             if score <= -0.05:
#                 return 'Negative'
#             elif score >= 0.05:
#                 return 'Positive'
#             else:
#                 return 'Neutral'
#         df2['Sentiment'] = df2['compound'].apply(getSentiment)

#         st.subheader("Sentiments of News Headlines, Last 5 Days")
#         color_discrete_map = {'Neutral': 'rgb(219,199,182)', 'Positive': 'rgb(219,199,182)', 'Negative':'rgb(219,199,182)'}
#         d = df2.groupby(by=["Sentiment"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
#         fig2=px.bar(d, x="Counts", y="Sentiment", color="Sentiment", color_discrete_map= color_discrete_map)
#         fig2.update_traces(textfont_size=14, textangle=90, textposition="inside", cliponaxis=False)
#         fig2.update_layout(template='simple_white',showlegend=False, autosize=False,width=650, height=400,margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=10,
#         pad=0
#         ))
#         st.write(fig2, use_container_width=True)

#     with img:
#         st.write(" ")
#         st.write(" ")
#         st.write(" ")
#         st.write(" ")
#         st.image("https://raw.githubusercontent.com/SergeBoyajian/Capstonespx/main/aditya-vyas-7ygsBEajOG0-unsplash.jpg", use_column_width=True)
#         # image = Image.open('C:\\Users\\admin\\Desktop\\AUB\\Capstone\\HD Pics\\aditya-vyas-7ygsBEajOG0-unsplash.jpg')
#         # st.image(image, use_column_width=True)
# ########################################################### Sentiment Analysis on Tweets (past 5 days)
if menu_id=="Tweets & Sentiments":
    col1, col2= st.columns(2)
# Scrape last 5 days tweets on S&P 500
    with col1:
    # #Initiate scrape for S&P 500 tweets
        with st.spinner("Scraping Last 5 Days Tweets..."):
            start= date.today() - timedelta(days=5)
            query=f' "S&P 500" -bitcoin -btc -crypto -doge -coin lang:en since:{start} -filter:replies'
            tweets= []
            limit=60000
            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
              # if len(tweets)==limit:
              #   break
              # else:
              tweets.append([tweet.date, tweet.username, tweet.content])

            df= pd.DataFrame(tweets, columns= ["Date", "User", "Tweet"])
        st.success("Done!")
        st.subheader("Raw Tweets")
        st.write(df)



# Conduct sentiment analysis on scraped tweets
        analyser = SentimentIntensityAnalyzer()
        # function to calculate polarity scores
        pol = lambda x: analyser.polarity_scores(x)
        # creating new column 'polarity' in clean_df
        df['polarity'] = df['Tweet'].apply(pol)
        df1=df['polarity'].apply(pd.Series)



        df2 = pd.concat([df, df['polarity'].apply(pd.Series)], axis=1)

        #Classify tweets based on their compound scores to three labels: Positive, Negative and Neutral
        def getSentiment(score):
            if score <= -0.05:
                return 'Negative'
            elif score >= 0.05:
                return 'Positive'
            else:
                return 'Neutral'
        df2['Sentiment'] = df2['compound'].apply(getSentiment)

        st.subheader("Sentiments of Tweets")
        color_discrete_map = {'Neutral': 'rgb(219,199,182)', 'Positive': 'rgb(219,199,182)', 'Negative':'rgb(219,199,182)'}
        d = df2.groupby(by=["Sentiment"]).size().reset_index(name="Counts").sort_values(by='Counts', ascending=False)
        fig2=px.bar(d, x="Counts", y="Sentiment", color="Sentiment", color_discrete_map= color_discrete_map)
        fig2.update_traces(textfont_size=14, textangle=90, textposition="inside", cliponaxis=False)
        fig2.update_layout(template='simple_white',showlegend=False, autosize=False,width=650, height=400,margin=dict(
        l=0,
        r=0,
        b=0,
        t=10,
        pad=0
        ))
        st.write(fig2, use_container_width=True)
# Input S&P price chart from TradingView
    with col2:
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.subheader("S&P 500 Dynamic Chart")
        # def main():

        html_temp = """
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
        <div id="tradingview_e067c"></div>
        <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/FXOPEN-SPXM/" rel="noopener" target="_blank"><span class="blue-text">SPXM Chart</span></a> by TradingView</div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget(
        {
        "width": 780,
        "height": 600,
        "symbol": "FXOPEN:SPXM",
        "interval": "D",
        "timezone": "Africa/Cairo",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_e067c"
        }
        );
        </script>
        </div>
        <!-- TradingView Widget END -->"""

        st.components.v1.html(html_temp, height=610, scrolling=False)
            # if __name__ == "__main__":
            #     main()
########################################################### LSTM Forecast on S&P 500 Price Chart
if menu_id=="Price Forecast":
    end_date= date.today()
    spx=yf.Ticker("^GSPC")
    df = spx.history(start= "2002-09-19", end=end_date, interval="1d")
    df.reset_index(inplace=True)

# Train the model on the WHOLE data available
    values_2 = df["Close"].values
    val_data_len = math.ceil(len(values_2))
    scaler = MinMaxScaler(feature_range=(0,1))
    #scaler= StandardScaler()
    scaled_data_2 = scaler.fit_transform(values_2.reshape(-1,1))

    x_all = []
    y_all = []

    for i in range(5, len(scaled_data_2)):
        x_all.append(scaled_data_2[i-5:i, 0])
        y_all.append(scaled_data_2[i, 0])

    x_all, y_all = np.array(x_all), np.array(y_all)
    x_val_2 = np.reshape(x_all, (x_all.shape[0], x_all.shape[1], 1))

    # x_train = []
    # y_train = []
    #
    # for i in range(5, len(scaled_data)):
    #     x_train.append(scaled_data[i-5:i, 0])
    #     y_train.append(scaled_data[i, 0])
    #
    # x_train, y_train = np.array(x_train), np.array(y_train)
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    # tf.random.set_seed(1)
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=False, input_shape=(x_all.shape[1], 1)))
    # model.add(layers.LSTM(60, return_sequences=False))
    # model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    with st.spinner("Tuned LSTM model is fitting all closing prices to date..."):
        model.fit(x_all, y_all, batch_size= 5, epochs=20)
    st.success("Done!")


    #Scrape the previous 5 business Days

    spx=yf.Ticker("^GSPC")
    start= date.today()-BDay(5)
    end= date.today()
    x_input = spx.history(start=start, end=end, interval="1d")
    x_input.reset_index(inplace=True)
    x_input= x_input["Close"].values
    print(x_input)
    scaler = MinMaxScaler(feature_range=(0,1))
    #scaler= StandardScaler()
    x_input = scaler.fit_transform(x_input.reshape(-1,1))
    x_input= np.array(x_input)
    x_input = np.reshape(x_input, (-1, 5, 1))


#Predict the upcoming 5 closing prices:
# demonstrate prediction for next 5 days
    temp_input=list(x_input)
    from numpy import array
    lst_output=[]
    n_steps=5
    i=0
    while(i<5):

        if(len(temp_input)>5):
            # tf.random.set_seed(1)
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            # tf.random.set_seed(1)
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            #print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    lst_output=scaler.inverse_transform(lst_output)
    # print(lst_output)

    st.subheader("LSTM 5 Days Forward Forecast")
#Plot Forecasts:
    # organize the results in a data frame
    df_past = df[["Close"]]
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=5)
    df_future['Forecast'] = lst_output.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('Date')
#Plot forecasts

    results= results.reset_index()#Plot S&P

    fig1= go.Figure()
    fig1.add_scatter(x=results["Date"] ,y=results['Actual'], name="Actual")

    fig1.add_scatter(x=results["Date"], y=results['Forecast'], mode='lines', name= "5d Forecast")
    fig1.update_layout( autosize=False,width=1600, height=800 )

    # Show plot
    st.write(fig1)
    st.write("Please zoom in on the upper right part of the chart to perceive the forecast")
