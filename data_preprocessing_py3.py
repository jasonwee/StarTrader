
import pandas as pd
import numpy as np
import random
import logging
#import talib as tb

from datetime import datetime, timedelta
#from feature_select import FeatureSelector

from typing import Iterable



# Start and end period of historical data in question
START_TRAIN = datetime(2000, 1, 1)
END_TRAIN = datetime(2017, 2, 12)
START_TEST = datetime(2017, 2, 12)
END_TEST = datetime(2019, 2, 22)



# DJIA component stocks
DJI = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX', 'UNH', 'VZ', 'WMT']

CONTEXT_DATA = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', '^VIX', 'GLD', '^TYX', '^TNX' , 'SHY', 'SHV']          

CONTEXT_DATA_N = ['S&P 500', 'Dow Jones Industrial Average', 'NASDAQ Composite', 'Russell 2000', 'SPDR S&P 500 ETF',
 'Invesco QQQ Trust', 'CBOE Volatility Index', 'SPDR Gold Shares', 'Treasury Yield 30 Years',
 'CBOE Interest Rate 10 Year T Note', 'iShares 1-3 Year Treasury Bond ETF', 'iShares Short Treasury Bond ETF']

random.seed(633)
RANDOM_STOCK = random.sample(DJI, 1)

#13 WEEK TREASURY BILL (^IRX)
# https://finance.yahoo.com/quote/%5EIRX?p=^IRX&.tsrc=fin-srch
RISK_FREE_RATE = ((1+0.02383)**(1.0/252))-1 # Assuming 1.43% risk free rate divided by 360 to get the daily risk free rate.

logger = logging.getLogger(__name__)


class DataRetrieval:

    def __init__(self):
        self._dji_components_data()

    def _get_daily_data(self, symbol : str) -> pd.core.frame.DataFrame:

        daily_price = pd.read_csv("{}{}{}".format('./data/', symbol, '.csv'), index_col='Date', parse_dates=True)
        
        return daily_price


    def _dji_components_data(self):
        """
        This function retrieve all components data and assembles the required 
        OHLCV (open, high low, close, volume) data into respective data
        """        

        for i in DJI + CONTEXT_DATA:
            #print(i)                             # MMM
            #print(DJI + CONTEXT_DATA_N)          # ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX', 'UNH', 'VZ', 'WMT', 'S&P 500', 'Dow Jones Industrial Average', 'NASDAQ Composite', 'Russell 2000', 'SPDR S&P 500 ETF', 'Invesco QQQ Trust', 'CBOE Volatility Index', 'SPDR Gold Shares', 'Treasury Yield 30 Years', 'CBOE Interest Rate 10 Year T Note', 'iShares 1-3 Year Treasury Bond ETF', 'iShares Short Treasury Bond ETF']
            #print((DJI + CONTEXT_DATA))          # ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX', 'UNH', 'VZ', 'WMT', '^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', '^VIX', 'GLD', '^TYX', '^TNX', 'SHY', 'SHV'] 
            #print((DJI + CONTEXT_DATA).index(i)) # 0
            #print((DJI + CONTEXT_DATA_N)[(DJI + CONTEXT_DATA).index(i)])  # MMM
            #print("Loading {}'s historical data".format((DJI + CONTEXT_DATA_N)[(DJI + CONTEXT_DATA).index(i)]))
            logger.info("Loading {}'s historical data".format((DJI + CONTEXT_DATA_N)[(DJI + CONTEXT_DATA).index(i)]))
            daily_price = self._get_daily_data(i)
            #print(daily_price.index)
            # DatetimeIndex(['2008-12-31', '2009-01-02', '2009-01-05', '2009-01-06',
            # '2009-01-07', '2009-01-08', '2009-01-09', '2009-01-12',
            # ...
            # dtype='datetime64[ns]', name='Date', length=2553, freq=None)
            if i == (DJI + CONTEXT_DATA)[0]:
                self.components_df_o = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                self.components_df_c = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                self.components_df_h = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                self.components_df_l = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                self.components_df_v = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                # Since this span more than 10 years of data, many corporate actions could have happened,
                # adjusted closing price is used instead
                self.components_df_o[i] = daily_price["Open"]
                self.components_df_c[i] = daily_price["Adj Close"]
                self.components_df_h[i] = daily_price["High"]
                self.components_df_l[i] = daily_price["Low"]
                self.components_df_v[i] = daily_price["Volume"]
            else:
                self.components_df_o[i] = daily_price["Open"]
                self.components_df_c[i] = daily_price["Adj Close"]
                self.components_df_h[i] = daily_price["High"]
                self.components_df_l[i] = daily_price["Low"]
                self.components_df_v[i] = daily_price["Volume"]

    def get_dailyprice_df(self):
        """
        Gets all stocks' close price and separates them into train and test set.
        """
        self.dow_stocks_test = self.components_df_c.loc[START_TEST:END_TEST][DJI]
        self.dow_stocks_train = self.components_df_c.loc[START_TRAIN:END_TRAIN][DJI]


    def get_all(self) -> tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
        """
        Response to external request to get all stock price in train and test set.
        """

        self.get_dailyprice_df()
        return self.dow_stocks_train, self.dow_stocks_test


    def technical_indicators_df(self, daily_data):
        """
        Assemble a dataframe of technical indicator series for a single stock
        """
        o = daily_data['Open'].values
        c = daily_data['Close'].values
        h = daily_data['High'].values
        l = daily_data['Low'].values
        v = daily_data['Volume'].astype(float).values
        # define the technical analysis matrix

        # Most data series are normalized by their series' mean
        ta = pd.DataFrame()
        ta['MA5'] = tb.MA(c, timeperiod=5) / tb.MA(c, timeperiod=5).mean()
        ta['MA10'] = tb.MA(c, timeperiod=10) / tb.MA(c, timeperiod=10).mean()
        ta['MA20'] = tb.MA(c, timeperiod=20) / tb.MA(c, timeperiod=20).mean()
        ta['MA60'] = tb.MA(c, timeperiod=60) / tb.MA(c, timeperiod=60).mean()
        ta['MA120'] = tb.MA(c, timeperiod=120) / tb.MA(c, timeperiod=120).mean()
        ta['MA5'] = tb.MA(v, timeperiod=5) / tb.MA(v, timeperiod=5).mean()
        ta['MA10'] = tb.MA(v, timeperiod=10) / tb.MA(v, timeperiod=10).mean()
        ta['MA20'] = tb.MA(v, timeperiod=20) / tb.MA(v, timeperiod=20).mean()
        ta['ADX'] = tb.ADX(h, l, c, timeperiod=14) / tb.ADX(h, l, c, timeperiod=14).mean()
        ta['ADXR'] = tb.ADXR(h, l, c, timeperiod=14) / tb.ADXR(h, l, c, timeperiod=14).mean()
        ta['MACD'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0] / \
                     tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0].mean()
        ta['RSI'] = tb.RSI(c, timeperiod=14) / tb.RSI(c, timeperiod=14).mean()
        ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0] / \
                         tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0].mean()
        ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1] / \
                         tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1].mean()
        ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2] / \
                         tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2].mean()
        ta['AD'] = tb.AD(h, l, c, v) / tb.AD(h, l, c, v).mean()
        ta['ATR'] = tb.ATR(h, l, c, timeperiod=14) / tb.ATR(h, l, c, timeperiod=14).mean()
        ta['HT_DC'] = tb.HT_DCPERIOD(c) / tb.HT_DCPERIOD(c).mean()
        ta["High/Open"] = h / o
        ta["Low/Open"] = l / o
        ta["Close/Open"] = c / o

        self.ta = ta

    def label(self, df : pd.core.frame.DataFrame, seq_length):
        return (df['Returns'] > 0).astype(int)

    def preprocessing(self, symbol):
        """
        Preprocess all stock data into a big dataframe of features with the help
        of a feature selector , also creates label data
        """
        print("\n")
        print("Preprocessing {} & its technical data".format(symbol))
        print("============================================")
        self.daily_data = pd.DataFrame()
        self.daily_data['Returns'] = pd.Series((self.components_df_c[symbol] / self.components_df_c[symbol].shift(1) - 1) * 100, index=self.components_df_c[symbol].index)
        self.daily_data['Open'] = self.components_df_o[symbol]
        self.daily_data['Close'] = self.components_df_c[symbol]
        self.daily_data['High'] = self.components_df_h[symbol]
        self.daily_data['Low'] = self.components_df_l[symbol]
        self.daily_data['Volume'] = self.components_df_v[symbol].astype(float)
        seq_length = 3
        self.technical_indicators_df(self.daily_data)
        self.X = self.daily_data[['Open', 'Close', 'High', 'Low', 'Volume']] / self.daily_data[['Open', 'Close', 'High', 'Low', 'Volume']].mean()
        self.y = self.label(self.daily_data, seq_length)
        X_shift = [self.X]

        for i in range(1, seq_length):
            shifted_df = self.daily_data[['Open', 'Close', 'High', 'Low', 'Volume']].shift(i)
            X_shift.append(shifted_df / shifted_df.mean())
        ohlc = pd.concat(X_shift, axis=1)
        ohlc.columns = sum([[c + 'T-{}'.format(i) for c in ['Open', 'Close', 'High', 'Low', 'Volume']] \
                            for i in range(seq_length)], [])
        self.ta.index = ohlc.index
        self.X = pd.concat([ohlc, self.ta], axis=1)
        self.Xy = pd.concat([self.X, self.y], axis=1)

        fs = FeatureSelector(data=self.X, labels=self.y)
        fs.identify_all(selection_params={'missing_threshold': 0.6,
                                          'correlation_threshold': 0.9,
                                          'task': 'regression',
                                          'eval_metric': 'auc',
                                          'cumulative_importance': 0.99})
        self.X_fs = fs.remove(methods='all', keep_one_hot=True)

        return self.X_fs
        
    def get_feature_dataframe(self, selected_stock):
        """
        Get the preprocessed dataframe and extract only the stocks in interest. 
        Returns a smaller dataframe
        """
        self.feature_df = pd.DataFrame()
        for s in selected_stock:
            if s == selected_stock[0]:
                df = self.preprocessing(s)
                df.columns = [str(s) + '_' + str(col) for col in df.columns]
                self.feature_df = df
            else:
                df = self.preprocessing(s)
                df.columns = [str(s) + '_' + str(col) for col in df.columns]
                self.feature_df = pd.concat([self.feature_df, df], axis=1)
        return self.feature_df

class MathCalc:
    """
    This class performs all the mathematical calculations
    """
    
    @staticmethod
    def calc_return(period : pd.core.series.Series) -> pd.core.series.Series:
        """
        This function computes the return of a series
        """
        period_return = period / period.shift(1) - 1
        return period_return[1:len(period_return)]


    @staticmethod
    def calc_monthly_return(series : pd.core.series.Series) -> pd.core.series.Series:
        """
        This function computes the monthly return
        
        https://stackoverflow.com/questions/17001389/pandas-resample-documentation
        M : month end
        """
        return MathCalc.calc_return(series.resample('ME').last())

    @staticmethod
    def positive_pct(series : pd.core.series.Series) -> pd.core.series.Series:
        """
        This function calculates the probably of positive values from a series of values.
        """
        return (float(len(series[series > 0])) / float(len(series)))*100

    @staticmethod
    def calc_yearly_return(series : pd.core.series.Series) -> pd.core.series.Series:
        """
        This function computes the yearly return
        
        https://stackoverflow.com/questions/17001389/pandas-resample-documentation
        AS : AS, YS    year start frequency
        """
        return MathCalc.calc_return(series.resample('YS').last())

    @staticmethod
    def max_drawdown(r : pd.Series | pd.core.frame.DataFrame) -> pd.core.series.Series:
        """
        This function calculates maximum drawdown occurs in a series of cummulative returns
        
        A drawdown is a peak-to-trough decline during a specific period for an 
        investment, trading account, or fund
        """
        dd = r.div(r.cummax()).sub(1)
        maxdd = dd.min()
        return round(maxdd, 2)

    @staticmethod
    def calc_lake_ratio(series: pd.Series) -> float:
        """
        This function computes lake ratio
        
        The lake ratio is a simple to understand ratio for use in measuring 
        performances of trading systems and indeed the trading account of a 
        trader or fund manager.
        """
        water = 0
        earth = 0
        series = series.dropna()
        water_level = []
        for i, s in enumerate(series):
            if i == 0:
                peak = s
            else:
                peak = np.max(series[0:i])
            water_level.append(peak)
            if s < peak:
                water = water + peak - s
            earth = earth + s
        return water / earth

    @staticmethod
    def calc_gain_to_pain(daily_series: pd.Series) -> float:
        """
        This function computes the gain to pain ratio given a series of cummulative returns
        
        I define the Gain to Pain ratio (GPR) as the sum of all monthly returns 
        divided by the absolute value of the sum of all monthly losses. This 
        performance measure indicates the ratio of cumulative net gain to the 
        cumulative loss realized to achieve that gain.
        """

        try:
            monthly_returns = MathCalc.calc_monthly_return(daily_series.dropna())
            sum_returns = monthly_returns.sum()
            sum_neg_months = abs(monthly_returns[monthly_returns < 0].sum())
            gain_to_pain = sum_returns / sum_neg_months if sum_neg_months != 0 else float('inf')
        except:
            gain_to_pain = 1.0

        #print(f"Gain to Pain ratio: {gain_to_pain}")
        return gain_to_pain

    @staticmethod
    def sharpe_ratio(returns : pd.Series | pd.core.frame.DataFrame) -> pd.core.series.Series:
        """
        Calculates Sharpe ratio from a series of returns.
        """
        return ((returns.mean() - RISK_FREE_RATE) / returns.std()) * np.sqrt(252)

    @staticmethod
    def downside_deviation(returns : pd.Series | pd.core.frame.DataFrame) -> np.float64:
        """
        This method returns a lower partial moment of the returns. Create an 
        array the same length as returns containing the minimum return threshold
        """
        target = 0
        df = pd.DataFrame(data=returns, columns=["Returns"], index=returns.index)
        df["Downside Returns"] = 0.0
        df.loc[df["Returns"] < target, "Downside Returns"] = df["Returns"] ** 2
        expected_return = df["Returns"].mean()

        return np.sqrt(df["Downside Returns"].mean())

    @staticmethod
    def sortino_ratio(returns : pd.Series | pd.core.frame.DataFrame) -> pd.core.series.Series:
        """
        Calculates Sortino ratio from a series of returns.
        
        The Sortino ratio measures the risk-adjusted return of an investment 
        asset, portfolio, or strategy.[1] It is a modification of the Sharpe 
        ratio but penalizes only those returns falling below a user-specified 
        target or required rate of return, while the Sharpe ratio penalizes 
        both upside and downside volatility equally. Though both ratios 
        measure an investment's risk-adjusted return, they do so in 
        significantly different ways that will frequently lead to differing 
        conclusions as to the true nature of the investment's return-generating 
        efficiency.
        """
        return ((returns.mean() - RISK_FREE_RATE) / MathCalc.downside_deviation(returns))* np.sqrt(252)

    @staticmethod
    def calc_kpi(portfolio : pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        """
        This function calculates individual portfolio KPI related its risk profile
        """

        kpi = pd.DataFrame(index=['KPI'], columns=['Avg. monthly return', 'Pos months pct', 'Avg yearly return',
                                                   'Max monthly dd', 'Max drawdown', 'Lake ratio', 'Gain to Pain',
                                                   'Sharpe ratio', 'Sortino ratio'])
        kpi.loc['Avg. monthly return', 0] = MathCalc.calc_monthly_return(portfolio['Total asset']).mean() * 100
        kpi.loc['Pos months pct', 0] = MathCalc.positive_pct(portfolio['Returns'])
        kpi.loc['Avg yearly return', 0] = MathCalc.calc_yearly_return(portfolio['Total asset']).mean() * 100
        kpi.loc['Max monthly dd', 0] = MathCalc.max_drawdown(MathCalc.calc_monthly_return(portfolio['CumReturns']))
        kpi.loc['Max drawdown', 0] = MathCalc.max_drawdown(MathCalc.calc_return(portfolio['CumReturns']))
        kpi.loc['Lake ratio', 0] = MathCalc.calc_lake_ratio(portfolio['Total asset'])
        kpi.loc['Gain to Pain', 0] = MathCalc.calc_gain_to_pain(portfolio['Total asset'])
        kpi.loc['Sharpe ratio', 0] = MathCalc.sharpe_ratio(portfolio['Returns'])
        kpi.loc['Sortino ratio', 0] = MathCalc.sortino_ratio(portfolio['Returns'])

        return kpi

    @staticmethod
    def colrow(i : int) -> tuple[int, int]:
        """
        This function calculate the row and columns index number based on the
        total number of subplots in the plot.

        Return:
             row: axis's row index number
             col: axis's column index number
        """

        # Do odd/even check to get col index number
        if i % 2 == 0:
            col = 0
        else:
            col = 1
        # Do floor division to get row index number
        row = i // 2

        return col, row


class Trading:
    """
    This class performs trading and all other functions related to trading
    """

    def __init__(self, dow_stocks_train : pd.core.frame.DataFrame, dow_stocks_test : pd.core.frame.DataFrame, dow_stocks_volume : pd.core.frame.DataFrame):
        self._dow_stocks_test = dow_stocks_test
        self.dow_stocks_train = dow_stocks_train
        self.daily_v = dow_stocks_volume
        self.remaining_stocks()

    def remaining_stocks(self):
        """
        This function finds out the remaining Dow component stocks after the
        selected stocks are taken.
        """
        dow_remaining = self._dow_stocks_test.drop(RANDOM_STOCK, axis=1)
        self.dow_remaining = [i for i in dow_remaining.columns]

    def find_non_correlate_stocks(self, num_non_corr_stocks):
        """
        This function performs trade with a portfolio starting with the number
        of stocks specified and find the required number of most uncorrelated
        stocks. Only the train set data is used to perform this task to avoid
        look ahead bias.
        """
        add_stocks = (min(num_non_corr_stocks, len(DJI))) - 1
        # Get the returns of the long only returns of all Dow component stocks during the pre-trading period.
        single_component_fund = SINGLE_TRADING_FUND
        share_distribution = single_component_fund / self.dow_stocks_train[RANDOM_STOCK].iloc[0]
        dow_stocks_values = self.dow_stocks_train[RANDOM_STOCK].mul(share_distribution, axis=1)
        portfolio_longonly_train = self.construct_book(dow_stocks_values, True)

        # find the most uncorrelated stocks with the one randomly selected
        # stock arranged from most uncorrelated to most correlated
        remaining_corr = self.stocks_corr(portfolio_longonly_train)

        # Assemble the non-correlate stocks
        ncs = RANDOM_STOCK

        adding_stocks = [i for i in remaining_corr[0:add_stocks].index]

        # add stocks to the random portfolio stock
        ncs = ncs + adding_stocks

        # Do buy and hold trade with a simple equally weighted portfolio of the 5 non-correlate stocks
        portfolio_values, portfolio_nc_5, kpi_nc_5 = self.diversified_trade(ncs, self.dow_stocks_train[ncs])
        return portfolio_nc_5, kpi_nc_5, ncs

    @staticmethod
    def commission(num_share, share_value):
        """
        This function computes commission fee of every trade
        https://www.interactivebrokers.com/en/index.php?f=1590&p=stocks1
        """
        trade_value = num_share * share_value
        max_comm_fee = 0.01 * trade_value
        comm_fee = 0.005 * num_share

        if max_comm_fee < comm_fee:
            comm_fee = max_comm_fee
        elif comm_fee <= max_comm_fee and comm_fee > 1.0:
            pass
        elif comm_fee < 1.0 and num_share > 0:
            comm_fee = 1.0
        elif num_share == 0:
            comm_fee = 0.0

        return comm_fee

    @staticmethod
    def slippage_price(price, stock_quantity, day_volume):
        """
        This function performs slippage price calculation using Zipline's volume share model
        https://www.zipline.io/_modules/zipline/finance/slippage.html
        """

        volumeShare = stock_quantity / float(day_volume)
        impactPct = volumeShare ** 2 * PRICE_IMPACT

        if stock_quantity > 0:
            slipped_price = price * (1 + impactPct)
        else:
            slipped_price = price * (1 - impactPct)

        return slipped_price
