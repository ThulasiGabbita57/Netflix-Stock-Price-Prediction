import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import streamlit as st
from plotly import graph_objs as go
import joblib


df = pd.read_csv('D:\\Fall2023\\DATA606\\Final_Project\\NFLX.csv')
#Webpage Title
st.title('Netflix Stock Price Prediction')
#Displaying the data
st.subheader('Data From 02/05/2018 to 02/04/2022')
st.write(df)

#Statistical Summary of the data
st.subheader('Statistical summary of the data')
st.write(df.describe())

#Visualizations
st.subheader('Visulaization of Netflix stock prices over 5 years')

# Plot Open
def plot_open():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name="stock_open"))
	fig.layout.update(title_text='Open price of Netflix', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
def plot_high():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['High'], name="stock_high"))
	fig.layout.update(title_text='High price of Netflix', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
def plot_low():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Low'], name="stock_low"))
	fig.layout.update(title_text='Low price of Netflix', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
def plot_close():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
	fig.layout.update(title_text='Close price of Netflix', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
def plot_adj_close():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], name="stock_adj_close"))
	fig.layout.update(title_text='Adj Close price of Netflix', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
# Plot raw data
def plot_volume():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume'], name="stock_volume"))
	fig.layout.update(title_text='Volume of Netflix', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_open()
plot_high()
plot_low()
plot_close()
plot_adj_close()
plot_volume()

st.subheader('Distribution of Close Price')
fig = plt.figure(figsize=(15,5))
sns.histplot(df['Close'])
st.pyplot(fig)

#Daily returns
df['Daily Returns'] = df['Adj Close'].pct_change()

st.subheader('Distribution of Daily returns percentange of Netflix')
fig=plt.figure(figsize=(15,5))
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
sns.histplot(data=df['Daily Returns'],bins=80)
ax1.set_xlabel('Daily Returns in Percentage')
ax1.set_ylabel('Percentage')
ax1.set_title("Distribution of daily returns percetage of Netflix stock")
st.pyplot(fig)

st.subheader('Daily returns of Netflix over 5 years')
fig = plt.figure(figsize=(15,5))
df['Daily Returns'].plot()
plt.xlabel('Date')
plt.ylabel('Percent')
plt.title('Daily returns for the Netflix stock between 02/05/18 to 02/04/22')
st.pyplot(fig)

#Cumulative returns
df['Cumulative Returns'] =  (df['Daily Returns']+1).cumprod()

st.subheader('Cumulative returns of Netflix from 2018 to 2022')
fig = plt.figure(figsize=(15,5))
df['Cumulative Returns'].plot()
plt.xlabel("Date")
plt.ylabel("Percentage")
plt.title('Cumulative returns of Netflix stock from 2018 to 2022')
st.pyplot(fig)

#plot comparision of moving average and Close price 
st.subheader('Plot comparision of moving average and Close price ')
fig = plt.figure(figsize=(15,5))
df['Close'].rolling(window=30).mean().plot(label='30 day moving avg')
df['Close'].plot(label='Close price')
plt.title('Comparision of Close price and moving average for Netflix stock from 2018 to 2022')
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig)

# Train test split
st.subheader('Train and Test Split')
X = df[['Open','High','Low','Volume']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
st.write(f'Training samples: {X_train.shape[0]:,}')
st.write(f'Test samples: {X_test.shape[0]:,}')

#Scaling
scaler = StandardScaler().fit_transform(X_train)

# # Load ARIMA Model
# model = joblib.load('DATA606_Netflix_ARIMA.pkl')

# #testing values
# st.subheader('Predicted testing values vs original Close price')
# df['prediction_arima']=model.predict(start=800,end=999)
# fig = plt.figure(figsize=(20,5))
# df[["Close","prediction_arima"]].plot()
# st.pyplot(fig)

#Arima model

df1 = df.groupby('Date')['Close'].sum().reset_index()
df1.Date=pd.to_datetime(df1.Date)
df1.set_index(['Date'],inplace=True)

#adfuller test
def adfuller_test(trends):
    result = adfuller(trends)
    labels = ['ADF Test Statistic','p-value','#Lags Used','#Observation Used']
    for value,label in zip(result,labels):
        print(label  + ': ' + str(value))
    if result[1]<=0.05:
        print('Strong evidence against the null hypothesis, Hence REJECT null hypothesis Ho indicating that the series is Stationary')
    else:
        print('week evidence against null hypothesis, time series has a unit root. So, accept H0 and the series is not stationary.')
#Differencing
diff1=df1-df1.shift(1)
diff1=diff1.dropna()
adfuller_test(diff1)

model=ARIMA(df['Close'],order=(6,1,6))
result=model.fit()

st.subheader('After applying ARIMA model')

#testing values

df['prediction_arima']=result.predict(start=800,end=999)

# fig = plt.figure(figsize=(20,5))
# df["Close"].plot()
# df['prediction_arima'].plot()
# st.pyplot(fig)


st.subheader('Visualizing Predicted values vs Original values')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'][900:998], y=df['Close'][900:998], name="stock_close"))
fig.add_trace(go.Scatter(x=df['Date'][900:998], y=df['prediction_arima'][900:998],fillcolor='red', name = "prediction_values_arima"))
plt.xlabel('Date')
plt.title('Predicted Values vs Original Price')
plt.ylabel('Price')
st.plotly_chart(fig)