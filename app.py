import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Carregar o dataset
data = pd.read_csv("GlobalTemperatures.csv", parse_dates=["dt"])
data.set_index("dt", inplace=True)

# Selecionar a série de temperatura desejada
temperatures = data["LandMaxTemperature"].dropna()

# Dividir dados em treino e teste (80% treino, 20% teste)
train_size = int(len(temperatures) * 0.8)
train, test = temperatures[:train_size], temperatures[train_size:]

# Ajustar o modelo ARIMA
model_arima = ARIMA(train, order=(2, 2, 2))  # Exemplo: AR(2), I(2), MA(2)
model_arima_fit = model_arima.fit()

# Fazer previsões para os dados de teste usando ARIMA
forecast_arima = model_arima_fit.forecast(steps=len(test))
mse_arima = mean_squared_error(test, forecast_arima)
print(f"Erro Quadrado Médio (MSE) com ARIMA: {mse_arima}")

# Ajustar o modelo SARIMA
seasonal_order = (1, 1, 1, 12)  # 'm' é 12 para dados mensais
model_sarima = SARIMAX(train, order=(2, 2, 2), seasonal_order=seasonal_order)
model_sarima_fit = model_sarima.fit()

# Fazer previsões para os dados de teste usando SARIMA
forecast_sarima = model_sarima_fit.forecast(steps=len(test))
mse_sarima = mean_squared_error(test, forecast_sarima)
print(f"Erro Quadrado Médio (MSE) com SARIMA: {mse_sarima}")

# Prever para os próximos 26 anos (312 meses)
forecast_steps = 312
forecast_future = model_sarima_fit.forecast(steps=forecast_steps)

# Criar um índice de datas para as previsões futuras
last_date = temperatures.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='ME')

# Converter a previsão em um DataFrame
forecast_future_df = pd.DataFrame(forecast_future, index=future_dates, columns=['Forecast'])

# Gráficos para visualizar os resultados
plt.figure(figsize=(12, 6))
plt.plot(temperatures, label="Temperaturas Históricas")
plt.plot(test.index, forecast_arima, label="Previsões ARIMA", color="orange")
plt.plot(test.index, forecast_sarima, label="Previsões SARIMA", color="red")
plt.plot(future_dates, forecast_future_df['Forecast'], label="Previsões Futuras SARIMA", color="green", linestyle='--')
plt.legend()
plt.title("Comparação das Previsões: ARIMA vs SARIMA")
plt.xlabel("Data")
plt.ylabel("Temperatura")
plt.show()