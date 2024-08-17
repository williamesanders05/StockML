# Stock<span style="color:green">ML</span>
StockML uses Machine Learning and the AlphaVantage Stock API to predict the closing price of any stock.
## How This Application works
### 1. The User Enters A Stock Symbol
![Front Page](https://i.imgur.com/Pf5CJS0.jpeg)
Once the user enters the webpage they will be prompted to enter a stock ticker and once they do so the prediction will start.
### 2. The Algorithm
This project uses a linear regression algorithm based on mean squared error gradient descent. For each prediction the app does 2000 iterations of gradient descent over the past 100 days of information for that symbol. This algorithm uses the Open, High, and Low of the day to predict what the stock will close at.
### 3. The Prediction
![Front Page](https://i.imgur.com/umwR9GZ.jpeg)
On this page the app outputs:
1. The Opening price of the stock
2. The current Highest price the stock has hit that day
3. The current Lowest price the stock has hit that day
4. The predicted Closing price of the stock
