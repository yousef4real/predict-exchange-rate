from mpl_toolkits.mplot3d.proj3d import transform
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

alamat = 'https://raw.githubusercontent.com/ataislucky/Data-Science/main/dataset/USD-INR_Weekly.csv'
data = pd.read_csv(alamat)
print(data.sample(11))

print(data.isna().sum())

data = data.dropna()

print(data.describe())

figure = px.line(data, x="Date",
                 y="Close",
                 title='Conversion Rate over the years (USD/kwd)')
figure.show()

data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
data['Year'] = data['Date'].dt.year
data["Month"] = data["Date"].dt.month
print(data.head())

growth = data.groupby('Year').agg({'Close': lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100})

fig = go.Figure()
fig.add_trace(go.Bar(x=growth.index,
                     y=growth['Close'],
                     name='Yearly Growth'))

fig.update_layout(title="Yearly Growth of Conversion Rate (USD/kwd)",
                  xaxis_title="Year",
                  yaxis_title="Growth (%)",
                  width=900,
                  height=600)

pio.show(fig)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# Calculate monthly growth
data['Growth'] = data.groupby(['Year', 'Month'])['Close']
transform(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)

# Group data by Month and calculate average growth
grouped_data = data.groupby('Month').mean().reset_index()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=grouped_data['Month'],
    y=grouped_data['Growth'],
    marker_color=grouped_data['Growth'],
    hovertemplate='Month: %{x}<br>Average Growth: %{y:.2f}%<extra></extra>'
))

fig.update_layout(
    title="Aggregated Monthly Growth of Conversion Rate (USD/kwd)",
    xaxis_title="Month",
    yaxis_title="Average Growth (%)",
    width=900,
    height=600
)

pio.show(fig)

result = seasonal_decompose(data["Close"], model='multiplicative', period=24)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(8, 6)
fig.show()

model = auto_arima(data['Close'], seasonal=True, m=52, suppress_warnings=True)
print(model.order)

from statsmodels.tools.sm_exceptions import ValueWarning

warnings.simplefilter('ignore', ValueWarning)

p, d, q = 2, 1, 0
model = SARIMAX(data["Close"], order=(p, d, q),
                seasonal_order=(p, d, q, 52))
fitted = model.fit()
print(fitted.summary())

predictions = fitted.predict(len(data), len(data) + 90)
print(predictions)

fig = go.Figure()

# Add training data line plot
fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Training Data',
    line=dict(color='blue')
))

# Add predictions line plot
fig.add_trace(go.Scatter(
    x=predictions.index,
    y=predictions,
    mode='lines',
    name='Predictions',
    line=dict(color='red')
))

fig.update_layout(
    title="Training Data VS Predictions",
    xaxis_title="Date",
    yaxis_title="Close",
    legend_title="Data",
    width=1000,
    height=600
)

pio.show(fig)
