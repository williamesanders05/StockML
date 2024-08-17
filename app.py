from flask import Flask, render_template, request
from alpha_vantage.timeseries import TimeSeries
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    ## Check if form was submitted
    if request.method == "POST":
        ticker = request.form["ticker"]
        #get daily and intraday data
        ts = TimeSeries(key='ZAW10ODA2OT1H0A8', output_format='pandas')
        daily, meta_daily = ts.get_daily(symbol=ticker, outputsize='compact')
        intra, meta_intra = ts.get_intraday(symbol=ticker, interval='5min', outputsize='compact')
        #get current high and low from intraday data
        max, min = getExtremes(intra)
        #initialize variables for gradient
        x, y, z, b = 0, 0, 0, 0
        L = np.float64(1.0e-8)
        epochs = 2000
        for i in range(epochs):
            # for testing purposes
            # if i % 100 == 0:
            #     print(f'Epoch {i}: x = {x}, y = {y}, z = {z}, b = {b}')
            x, y, z, b = gradient_descent(x, y, z, b, daily, L)
        # calculate close price
        close = intra.iloc[-1]['1. open'] * x + max * y + min * z + b
        # pass variables to html template
        return render_template("symbol.html", ticker = ticker, close = close, open = intra.iloc[-1]['1. open'], high = max, low = min)
    return render_template("index.html")

def getExtremes(data):
    max = 0
    min = data.iloc[0]['3. low']
    for index, row in data.iterrows():
        if row['2. high'] > max:
            max = row['2. high']
        if row['3. low'] < min:
            min = row['3. low']
    return max, min

def gradient_descent(x_now, y_now, z_now, b_now, points, L):
    # initialize gradients
    open_gradient, high_gradient, low_gradient, b_gradient = 0, 0, 0, 0
        
    n = len(points)
        
    # loop over each test point and compute gradients based on the mean squared error of each variable
    for i in range(n):
        x = points.iloc[i]['1. open']
        y = points.iloc[i]['2. high']
        z = points.iloc[i]['3. low']
        result = points.iloc[i]['4. close']
            
        open_gradient += -(2/n) * x * (result - (x_now * x + y_now * y + z_now * z + b_now))
        high_gradient += (-2/n) * y * (result - (x_now * x + y_now * y + z_now * z + b_now))
        low_gradient += (-2/n) * z * (result - (x_now * x + y_now * y + z_now * z + b_now))
        b_gradient += (-2/n) * (result - (x_now * x + y_now * y + z_now * z + b_now))
            
    # apply learning rate and gradient to update weights
    new_x = x_now - open_gradient * L
    new_y = y_now - high_gradient * L
    new_z = z_now - low_gradient * L
    new_b = b_now - b_gradient * L
        
    return new_x, new_y, new_z, new_b
