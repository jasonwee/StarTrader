import unittest
import pytest
import pandas as pd

from data_preprocessing_py3 import MathCalc
from datetime import datetime

class MathCalcTest(unittest.TestCase):

    @pytest.mark.task(taskno=1)
    def test_calc_return(self):
        input_data = [
                      {'a': 1, 'b': 2, 'c': 3}
                     ]
        result_data = [[1.0, 0.5]]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                item_input = pd.Series(data=item, index=['a', 'b', 'c'])
                actual_result = list(MathCalc.calc_return(item_input))
                error_message = (f'Called MathCalc.calc_return({item}). '
                                f'The function returned "{actual_result}", but '
                                f'the tests expected "{expected}" as the Series.')

                self.assertEqual(actual_result, expected, msg=error_message)
    
    @pytest.mark.task(taskno=2)
    def test_calc_lake_ratio(self):
        input_data = [
                      {'a': 1, 'b': 2, 'c': 3}
                     ]
        result_data = [0.0]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                item_input = pd.Series(data=item, index=['a', 'b', 'c'])
                actual_result = MathCalc.calc_lake_ratio(item_input)
                error_message = (f'Called MathCalc.calc_lake_ratio({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the float.')

                self.assertEqual(actual_result, expected, msg=error_message)

    @pytest.mark.task(taskno=3)
    def test_calc_gain_to_pain(self):
        input_data = [ pd.Series(range(180), index=pd.date_range('1/1/2000', periods=180, freq='D')) ]

        result_data = [float("inf")]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                actual_result = MathCalc.calc_gain_to_pain(item)
                error_message = (f'Called MathCalc.calc_gain_to_pain({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the float.')

                self.assertEqual(actual_result, expected, msg=error_message)

    @pytest.mark.task(taskno=5)
    def test_downside_deviation(self):
        input_data = [ pd.DataFrame({  'Returns': [0.1, -0.2, 0.3, -0.4], }, index=['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04'])
                     ]
        result_data = [0.223606797749979]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                actual_result = MathCalc.downside_deviation(item)
                error_message = (f'Called MathCalc.downside_deviation({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the np.float64.')

                self.assertEqual(actual_result, expected, msg=error_message)

    @pytest.mark.task(taskno=6)
    def test_sortino_ratio(self):
        input_data = [ pd.DataFrame({  'Returns': [0.1, -0.2, 0.3, -0.4], }, index=['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']) ]

        result_data = [[-3.5562827808583224]]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                actual_result = MathCalc.sortino_ratio(item).to_list()
                error_message = (f'Called MathCalc.sortino_ratio({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the pd.Series.')

                self.assertEqual(actual_result, expected, msg=error_message)
                
    @pytest.mark.task(taskno=7)
    def test_sharpe_ratio(self):
        input_data = [ pd.DataFrame({'Returns': [0.1, -0.2, 0.3, -0.4], }, index=['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']) ]

        result_data = [[-2.5576606246889315]]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                actual_result = MathCalc.sharpe_ratio(item).to_list()
                error_message = (f'Called MathCalc.sharpe_ratio({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the pd.Series.')

                self.assertEqual(actual_result, expected, msg=error_message)

    @pytest.mark.task(taskno=8)
    def test_max_drawdown(self):
        input_data = [ pd.DataFrame({'Returns': [0.1, -0.2, 0.3, -0.4], }, index=['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']) ]

        result_data = [[-3.0]]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                actual_result = MathCalc.max_drawdown(item).to_list()
                error_message = (f'Called MathCalc.max_drawdown({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the pd.Series.')

                self.assertEqual(actual_result, expected, msg=error_message)

    @pytest.mark.task(taskno=9)
    def test_calc_yearly_return(self):
        input_data = [ pd.Series(range(9), index=pd.date_range('1/1/2000', periods=9, freq='min')) ]

        result_data = [[]]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                actual_result = MathCalc.calc_yearly_return(item).to_list()
                error_message = (f'Called MathCalc.calc_yearly_return({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the pd.Series.')

                self.assertEqual(actual_result, expected, msg=error_message)

    @pytest.mark.task(taskno=10)
    def test_calc_monthly_return(self):
        input_data = [ pd.Series(range(9), index=pd.date_range('1/1/2000', periods=9, freq='min')) ]

        result_data = [[]]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                actual_result = MathCalc.calc_monthly_return(item).to_list()
                error_message = (f'Called MathCalc.calc_monthly_return({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the pd.Series.')

                self.assertEqual(actual_result, expected, msg=error_message)

    @pytest.mark.task(taskno=11)
    def test_positive_pct(self):
        input_data = [ pd.Series(range(9), index=pd.date_range('1/1/2000', periods=9, freq='min')) ]

        result_data = [88.88888888888889]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                actual_result = MathCalc.positive_pct(item)
                error_message = (f'Called MathCalc.positive_pct({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the float.')

                self.assertEqual(actual_result, expected, msg=error_message)

    @pytest.mark.task(taskno=12)
    def test_calc_kpi(self):
        """
        input_data = [ pd.Series(range(9), index=pd.date_range('1/1/2000', periods=9, freq='min')) ]

        result_data = [88.88888888888889]

        for variant, (item, expected) in enumerate(zip(input_data, result_data), start=1):
            with self.subTest(f'variation #{variant}', item=item, expected=expected):
                actual_result = MathCalc.calc_kpi(item)
                error_message = (f'Called MathCalc.calc_kpi({item}). '
                                 f'The function returned {actual_result}, but the '
                                 f'tests expected {expected} as the float.')

                self.assertEqual(actual_result, expected, msg=error_message)
        """
        trading_book = pd.DataFrame(index=[datetime(2008, 12, 31)], columns=["Cash balance", "Portfolio value", "Total asset", "Returns", "Cum Returns"])
        trading_book["Cash balance"] = [100000]
        trading_book["Portfolio value"] = [0.0]
        trading_book["Total asset"] = [100000]
        trading_book["Returns"] = trading_book["Total asset"] / trading_book["Total asset"].shift(1) - 1
        trading_book["CumReturns"] = trading_book["Returns"].add(1).cumprod().fillna(1)
        
        actual_result = MathCalc.calc_kpi(trading_book)
        
        expected = '{"Avg. monthly return":{"KPI":null,"Avg. monthly return":null,"Pos months pct":null,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":null,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null},"Pos months pct":{"KPI":null,"Avg. monthly return":null,"Pos months pct":null,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":null,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null},"Avg yearly return":{"KPI":null,"Avg. monthly return":null,"Pos months pct":null,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":null,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null},"Max monthly dd":{"KPI":null,"Avg. monthly return":null,"Pos months pct":null,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":null,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null},"Max drawdown":{"KPI":null,"Avg. monthly return":null,"Pos months pct":null,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":null,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null},"Lake ratio":{"KPI":null,"Avg. monthly return":null,"Pos months pct":null,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":null,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null},"Gain to Pain":{"KPI":null,"Avg. monthly return":null,"Pos months pct":null,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":null,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null},"Sharpe ratio":{"KPI":null,"Avg. monthly return":null,"Pos months pct":null,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":null,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null},"Sortino ratio":{"KPI":null,"Avg. monthly return":null,"Pos months pct":null,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":null,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null},"0":{"KPI":null,"Avg. monthly return":null,"Pos months pct":0.0,"Avg yearly return":null,"Max monthly dd":null,"Max drawdown":null,"Lake ratio":0.0,"Gain to Pain":null,"Sharpe ratio":null,"Sortino ratio":null}}'
        
        self.assertEqual(actual_result.to_json(), expected, msg="check output")

