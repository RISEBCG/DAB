import pandas as pd
import numpy as np
import statsmodels.api as sm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

order = [2,1,2]
seasonal_order = [2,2,2,7]

def decompose(ts):
    # Perform time series decomposition using the additive model
    decomposition = sm.tsa.seasonal_decompose(ts, model='additive')

    # Extract the trend, seasonality, and residuals components
    trend = decomposition.trend
    seasonality = decomposition.seasonal
    residuals = decomposition.resid

    # Create traces for each component
    original_trace = go.Scatter(x=ts.index, y=ts, name='Original Time Series')
    trend_trace = go.Scatter(x=trend.index, y=trend.values, name='Trend Component')
    seasonality_trace = go.Scatter(x=seasonality.index, y=seasonality.values, name='Seasonality Component')
    residuals_trace = go.Scatter(x=residuals.index, y=residuals.values, name='Residuals Component')

    # Create subplot figure
    fig = make_subplots(rows=4, cols=1, subplot_titles=(
    'Original Time Series', 'Trend Component', 'Seasonality Component', 'Residuals Component'))

    # Add traces to subplot figure
    fig.add_trace(original_trace, row=1, col=1)
    fig.add_trace(trend_trace, row=2, col=1)
    fig.add_trace(seasonality_trace, row=3, col=1)
    fig.add_trace(residuals_trace, row=4, col=1)

    # Update layout of subplot figure
    fig.update_layout(height=800, title_text="Time Series Decomposition using Additive Model", xaxis=dict(type='date'))

    # Display plot
    fig.show()

def create_training_testing_data(df, cut_off_date):
    train = df[df.index < cut_off_date]
    test = df[df.index >= cut_off_date]

    print("Train data time range: {} - {}".format(train.index.min().strftime("%Y-%m-%d"), train.index.max().strftime("%Y-%m-%d")))
    print("Test data time range: {} - {}".format(test.index.min().strftime("%Y-%m-%d"), test.index.max().strftime("%Y-%m-%d")))

    return train, test

def train_forecasting_model(df, type='sarima'):
    if type == 'sarima':
        return sm.tsa.statespace.SARIMAX(np.log(df['DailySales']),
                                         order=order,
                                         seasonal_order=seasonal_order,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False,
                                         freq='D').fit()
    elif type == 'sarimax':
        return sm.tsa.statespace.SARIMAX(np.log(df['DailySales']),
                                         order=order,
                                         seasonal_order=seasonal_order,
                                         exog=df.drop(['DailySales'], axis=1).dropna(),
                                         enforce_stationarity=False,
                                         enforce_invertibility=False,
                                         freq='D').fit()
    else:
        raise Exception("Wrong Model Type: It has to be either sarima or sarimax")


def evaluate_model_performance(test, model, type='sarima'):

    start_index = test.index.min()
    end_index = test.index.max()

    excel_forecast = pd.read_csv('https://raw.githubusercontent.com/RISEBCG/DAB/main/M3A3-forecasting/ffc_sales_excel_predict.csv', parse_dates=['Date'], index_col=['Date'])
    excel_mae = mean_absolute_error(y_true=test['DailySales'], y_pred=excel_forecast['Daily Sales'])
    excel_mse = mean_squared_error(y_true=test['DailySales'], y_pred=excel_forecast['Daily Sales'])

    if (type != 'sarima') and (type != 'sarimax'):
        raise Exception("Wrong Model Type: It has to be either sarima or sarimax")

    elif type == 'sarima':
        forecast = model.predict(start=start_index, end=end_index)
        color = 'red'
    else:
        forecast = model.predict(start=start_index, end=end_index, exog=test.drop('DailySales', axis=1).dropna())
        color = 'yellow'

    forecast = np.exp(pd.Series(forecast, index=test.index))
    mae = mean_absolute_error(y_true=test['DailySales'], y_pred=forecast)
    mse = mean_squared_error(y_true=test['DailySales'], y_pred=forecast)

    print("{} | MAE: {:.3f} | MSE: {:.3f}".format(type.upper(), mae, mse))
    print("Excel | MAE: {:.3f} | MSE: {:.3f}".format(excel_mae, excel_mse))

    observed_trace = go.Scatter(x=test.index, y=test['DailySales'].tolist(), name='Observed (test data)')
    sarima_trace = go.Scatter(x=forecast.index, y=forecast.tolist(), name='{}'.format(type.upper()), line=dict(color=color))
    excel_trace = go.Scatter(x=excel_forecast.index, y=excel_forecast['Daily Sales'], name='Excel', line=dict(color='green'))

    # Create subplot figure
    fig = go.Figure()

    # Add traces to subplot figure
    fig.add_trace(observed_trace)
    fig.add_trace(sarima_trace)
    fig.add_trace(excel_trace)

    # Update layout of subplot figure
    fig.update_layout(title='{} Model vs Excel Model Performance'.format(type.upper()), xaxis_title='Date', yaxis_title='Value')

    # Display plot
    fig.show()

    return {
        'mae': mae,
        'mse': mse,
        'excel_mae': excel_mae,
        'excel_mse': excel_mse,
        'model_trace': sarima_trace,
        'excel_trace': excel_trace,
        'trace': observed_trace
    }

def combine_viz(sarima_outputs, sarimax_outputs):
    print("SARIMA | MAE: {:.3f} | MSE: {:.3f}".format(sarima_outputs['mae'], sarima_outputs['mse']))
    print("SARIMAX | MAE: {:.3f} | MSE: {:.3f}".format(sarimax_outputs['mae'], sarimax_outputs['mse']))
    print("Excel | MAE: {:.3f} | MSE: {:.3f}".format(sarima_outputs['excel_mae'], sarima_outputs['excel_mse']))

    # Create subplot figure
    fig = go.Figure()

    fig.add_trace(sarima_outputs['trace'])
    fig.add_trace(sarima_outputs['model_trace'])
    fig.add_trace(sarimax_outputs['model_trace'])
    fig.add_trace(sarima_outputs['excel_trace'])

    # Update layout of subplot figure
    fig.update_layout(title='Model Performance Comparison', xaxis_title='Date', yaxis_title='Value')

    # Display plot
    fig.show()




