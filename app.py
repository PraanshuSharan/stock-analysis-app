import logging
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the dataset
global_stock_data = pd.read_csv('sample_stock_prices.csv', index_col=0, parse_dates=True)

# Function to create and train LSTM model
def create_and_train_lstm_model(historical_data):
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(historical_data.reshape(-1, 1))
    
    # Create training data
    x_train = []
    y_train = []
    for i in range(60, len(scaled_data) - 7):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i:i + 7, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=7))  # Output 7 units for 7-day prediction
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    return model, scaler



# Route for the main page
@app.route('/')
def index():
    stock_names = global_stock_data.columns.tolist()
    return render_template('index.html', stock_names=stock_names)

# Route for plotting stock trends
@app.route('/plot', methods=['POST'])
def plot():
    logger.debug("Plot route called")
    data = request.json
    period = data['period']
    company1 = data['company1']
    company2 = data.get('company2')
    show_daily_avg = data['show_daily_avg']

    # Validation
    if company1 not in global_stock_data.columns:
        logger.error("Invalid company name: %s", company1)
        return jsonify({'error': f"Invalid company name. Please select from the following: {', '.join(global_stock_data.columns)}"}), 400
    if company2 and company2 not in global_stock_data.columns:
        logger.error("Invalid company name: %s", company2)
        return jsonify({'error': f"Invalid company name. Please select from the following: {', '.join(global_stock_data.columns)}"}), 400

    # Filtering Period
    end_date = global_stock_data.index[-1]
    if period == '1D':
        start_date = end_date - pd.DateOffset(days=1)
    elif period == '1W':
        start_date = end_date - pd.DateOffset(weeks=1)
    elif period == '1M':
        start_date = end_date - pd.DateOffset(months=1)
    elif period == 'Lifetime':
        start_date = global_stock_data.index[0]
    else:
        start_date = end_date - pd.DateOffset(weeks=1)

    # Filtered data
    filtered_data = global_stock_data[start_date:end_date]

    daily_average = filtered_data.mean(axis=1)

    # Plot 
    plt.figure(figsize=(14, 8))
    if show_daily_avg:
        plt.plot(filtered_data.index, daily_average, label='Daily Average', linestyle='--', color='black')
    plt.plot(filtered_data.index, filtered_data[company1], label=company1)
    if company2:
        plt.plot(filtered_data.index, filtered_data[company2], label=company2)
    plt.title(f'Stock Prices for {company1} {f"and {company2}" if company2 else ""} from {start_date.date()} to {end_date.date()}', fontsize=14, loc='center')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    logger.debug("Plot route processed successfully")
    return jsonify({'image': image_base64})

# Route for predicting stock prices
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    logger.debug("Predict route called")
    data = request.json
    company = data['company']

    # Validation
    if company not in global_stock_data.columns:
        logger.error("Invalid company name: %s", company)
        return jsonify({'error': f"Invalid company name. Please select from the following: {', '.join(global_stock_data.columns)}"}), 400

    historical_data = global_stock_data[company].values
    logger.debug(f"Historical data for {company}: {historical_data}")

    model, scaler = create_and_train_lstm_model(historical_data)

    # Prepare data for prediction
    last_60_days = historical_data[-60:]
    scaled_last_60_days = scaler.transform(last_60_days.reshape(-1, 1))
    x_test = [scaled_last_60_days]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    predicted_prices = predicted_prices.flatten().tolist()
    
    dates = pd.date_range(start=global_stock_data.index[-1] + pd.DateOffset(1), periods=7).tolist()

    logger.debug(f"Predicted prices: {predicted_prices}")
    return jsonify({'dates': dates, 'prices': predicted_prices})


@app.route('/ranking', methods=['POST'])
def ranking():
    logger.debug("Ranking route called")
    data = request.json
    period = data['period']

    # Filtering Period
    end_date = global_stock_data.index[-1]
    if period == '1D':
        start_date = end_date - pd.DateOffset(days=1)
    elif period == '1W':
        start_date = end_date - pd.DateOffset(weeks=1)
    elif period == '1M':
        start_date = end_date - pd.DateOffset(months=1)
    elif period == 'Lifetime':
        start_date = global_stock_data.index[0]
    else:
        start_date = end_date - pd.DateOffset(weeks=1)

    # Filtered data
    filtered_data = global_stock_data[start_date:end_date]

    # Calculate average price over the period
    average_prices = filtered_data.mean().sort_values(ascending=False)

    rankings = [{'name': stock, 'average_price': avg_price} for stock, avg_price in average_prices.items()]

    logger.debug("Ranking route processed successfully")
    return jsonify({'rankings': rankings})



if __name__ == '__main__':
    app.run(debug=True)
