import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Load the data
data = pd.read_csv("KEN_GBR_USD_2010M1_2023M10.csv")

#Calculate the actual ∆cpit series
data["Exchange Rate Change"] = data["Exchange Rate"].diff()

#Select the appropriate sample
sample = data[(data["Year"] >= 2010) & (data["Month"] <= 12)]

#Define the ARIMA model
model = sm.tsa.statespace.SARIMAX(sample["Exchange Rate"], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))

#Estimate the model
results = model.fit()

#Generate forecasts
start = pd.to_datetime("2022-01-01")
end = pd.to_datetime("2023-08-31")
forecasts = results.forecast(start, end)

#Plot the actual and forecast series
plt.plot(sample["Date"], sample["Exchange Rate Change"], label="Actual")
plt.plot(forecasts.index, forecasts, label="Forecast")
plt.legend()
plt.show()

