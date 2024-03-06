[//]: # (Image References)

[image1]: https://github.com/jiewwantan/StarTrader/blob/master/train_iterations_9.gif "Training iterations"
[image2]: https://github.com/jiewwantan/StarTrader/blob/master/test_iteration_1.gif "Testing trained model with one iteration"
[image3]: https://github.com/jiewwantan/StarTrader/blob/master/test_result/portfolios_return.png "Trading strategy performance returns comparison"
[image4]: https://github.com/jiewwantan/StarTrader/blob/master/test_result/portfolios_risk.png "Trading strategy performance risk comparison"

# **StarTrader:** <br />Intelligent Trading Agent Development<br /> with Deep Reinforcement Learning

### Introduction

This project sets to create an intelligent trading agent and a trading environment that provides an ideal learning ground. A real-world trading environment is complex with stock, related instruments, macroeconomic, news and possibly alternative data in consideration. An effective agent must derive efficient representations of the environment from high-dimensional input, and generalize past experience to new situation.  The project adopts a deep reinforcement learning algorithm, deep deterministic policy gradient (DDPG) to trade a portfolio of five stocks. Different reward system and hyperparameters was tried. Its performance compared to models created by recurrent neural network, modern portfolio theory, simple buy-and-hold and benchmark DJIA index. The agent and environment will then be evaluated to deliberate possible improvement and the agent potential to beat professional human trader, just like Deepmind’s Alpha series of intelligent game playing agents.

The trading agent will learn and trade in [OpenAI Gym](https://gym.openai.com/) environment. Two Gym environments are created to serve the purpose, one for training (StarTrader-v0), another testing (StarTraderTest-v0). Both versions of StarTrader will utilize Gym's baseline implmentation of Deep deterministic policy gradient (DDPG). 

A portfolio of five stocks (out of 27 Dow Jones Industrial Average stocks) are selected based on non-correlation factor. StarTrader will trade these five non-correlated stocks by learning to maximize total asset (portfolio value + current account balance) as its goal. During the trading process, StarTrader-v0 will also optimize the portfolio by deciding how many stock units to trade for each of the five stocks.

Based on non-correlation factor, a portfolio optimization algorithm has chosen the following five stocks to trade: 

1. American Express
2. Wal Mart
3. UnitedHealth Group
4. Apple
5. Verizon Communications
		
The preprocessing function creates technical data derived from each of the stock’s OHLCV data. On average there are roughly 6-8 time series data derived for each stock. 

Apart from stock data, context data is also used to aid learning: 

1. S&P 500 index
2. Dow Jones Industrial Average index
3. NASDAQ Composite index
4. Russell 2000 index 
5. SPDR S&P 500 ETF
6. Invesco QQQ Trust
7. CBOE Volatility Index
8. SPDR Gold Shares 
9. Treasury Yield 30 Years
10. CBOE Interest Rate 10 Year T Note 
11. iShares 1-3 Year Treasury Bond ETF
12. iShares Short Treasury Bond ETF

Similarly, technical data derived from the above context data’s OHLCV data are being created. All data preprocessing is handled by two modules:
1. data_preprocessing.py
2. feature_select.py 

The preprocessed data are then being fed directly to StarTrader’s trading environment: class StarTradingEnv. 

The feature selection module (feature_select.py) select about 6-8 features out of 41 OHLCV and its technical data, In total, there are 121 features (may varies on different machine as the algorithm is not seeded) with about 36 stock feature data and the rest are context feature data. 

When trading is executed, 121 features along with total asset, current asset holdings and unrealized profit and loss will form a complete state space for the agent to trade and learn. The state space is designed to allow the agent to get a sense of the instantaneous environment in addition to how its interactions with the environment affects future state space. In another words, the trading agent bears the fruits and consequences of its own actions. 

### Training agent on 9 iterations
![Training iterations][image1]

### Testing agent on one iteration 
No learning or model refinement, purely on testing the trained model. 
Trading agent survived the major market correction in 2018 with 1.13 Sharpe ratio. <br />

![Testing trained model with one iteration][image2]

### Compare agent's performance with other trading strategies
DDPG is the best performer in terms of cumulative returns. However with a much less volatile ride, RNN-LSTM model has better risk-adjusted return: the highest Sharpe ratio (1.88) and Sortino ratio (3.06). Both RNN-LSTM and DRL-DDPG modelled trading strategies have trading costs: commission (based on Interactive Broker's fee) and slippage (modelled by Zipline and based on stock's daily volume) incorporated since there are many transactions during the trading window. The other buy-and-hold strategies' trading costs are omitted since there is stocks are only transacted once. 
DDPG's reward system shall be modified to yield higher risk-adjusted return. 
For a fair comparison, LSTM model uses the same training data and similar backtester as DDPG model.

![Trading strategy performance returns comparison][image3]
![Trading strategy performance risk comparison][image4]




## Installation instructions:

[debian12](./install-debian12.md)
