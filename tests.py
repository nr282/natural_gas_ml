import unittest
from weather import get_all_temperature_data
import datetime
from natural_gas_price import load_natural_gas_prices
from training import (TimeSeries, TimeSeriesDataframe,
                      FeatureDataframe, develop_complete_dataframe,
                      train_model_with_logistic_regression,
                      filter_dataframe)
import pandas as pd
import math
import numpy as np
from training import split_dataset, run_classification_model

class TestNaturalGasDeepLearning(unittest.TestCase):

    def setUp(self):

        time_index = pd.Index([datetime.datetime(2014, 1, 1),
                               datetime.datetime(2014, 1, 2),
                               datetime.datetime(2014, 1, 3),
                               datetime.datetime(2014, 1, 4),
                               datetime.datetime(2014, 1, 5)])

        ser_1 = pd.Series([1, 2, 3, 4, 5], index=time_index, name="Test_Value_1")

        time_index = pd.Index([datetime.datetime(2014, 1, 1),
                               datetime.datetime(2014, 1, 2),
                               datetime.datetime(2014, 1, 3),
                               datetime.datetime(2014, 1, 4),
                               datetime.datetime(2014, 1, 5)])

        ser_2 = pd.Series([1, 2, 3, 4, 5], index=time_index, name="Test_Value_2")

        time_index = pd.Index([datetime.datetime(2014, 1, 4),
                               datetime.datetime(2014, 1, 5),
                               datetime.datetime(2014, 1, 7),
                               datetime.datetime(2014, 1, 10),
                               datetime.datetime(2014, 1, 11)])

        ser_3 = pd.Series([1, 2, 3, 4, 5], index=time_index, name="Test_Value_3")

        self.indexes = [ser_1, ser_2]
        self.ser_1 = ser_1
        self.ser_2 = ser_2
        self.ser_3 = ser_3



    def test_weather_data(self):
        start = datetime.datetime(2018, 1, 1)
        end = datetime.datetime(2018, 12, 31)
        df = get_all_temperature_data(start, end)
        self.assertTrue(len(df) > 0)
        self.assertTrue(len(df.City.unique()) > 0)
        self.assertTrue(len(df.State.unique()) > 0)


    def test_read_csv_file_from_file(self):

        df = load_natural_gas_prices()
        self.assertTrue(len(df) > 0)
        self.assertTrue("Month" in df.columns)
        self.assertTrue("Henry Hub Natural Gas Spot Price Dollars per Million Btu" in df.columns)

    def test_time_series_object(self):

        ts = TimeSeries(self.ser_1)
        s = ts.size
        start_date = ts.start_date
        end_date = ts.end_date

        self.assertTrue(s > 0)
        self.assertTrue(type(start_date) == pd.Timestamp)
        self.assertTrue(ts.name == "Test_Value_1")

    def test_time_series_dataframe(self):


        ts_1 = TimeSeries(self.ser_1)
        ts_2 = TimeSeries(self.ser_2)
        ts_df = TimeSeriesDataframe([ts_1, ts_2])
        self.assertTrue(type(ts_df.get_dataframe()) == pd.DataFrame)
        self.assertTrue(len(ts_df.df) > 0)
        self.assertTrue(all([column in ts_df.df.columns for column in ["Test_Value_1", "Test_Value_2"]]))


    def test_performance_time_series_dataframe(self):
        """
        Performance testing of 10 million time series.

        30 Seconds For Ten Million Time Series


        :return:
        """


        ts_1 = TimeSeries(self.ser_1)
        n = 1000000
        ts_df = TimeSeriesDataframe([ts_1 for _ in range(n)])
        self.assertTrue(type(ts_df.get_dataframe()) == pd.DataFrame)
        self.assertTrue(len(ts_df.df) > 0)
        self.assertTrue(all([column in ts_df.df.columns for column in ["Test_Value_1"]]))

    def test_time_series_dataframe_with_non_homogenous_time_series(self):
        #What is the goal here.


        ts_1 = TimeSeries(self.ser_1)
        ts_2 = TimeSeries(self.ser_2)
        ts_3 = TimeSeries(self.ser_3)
        ts_df = TimeSeriesDataframe([ts_1, ts_2, ts_3])
        self.assertTrue(type(ts_df.get_dataframe()) == pd.DataFrame)
        self.assertTrue(len(ts_df.df) > 0)
        self.assertTrue(all([column in ts_df.df.columns for column in ["Test_Value_1", "Test_Value_2", "Test_Value_3"]]))
        self.assertTrue(math.isnan(ts_df.df["Test_Value_1"].iloc[-1]))
        self.assertTrue(math.isnan(ts_df.df["Test_Value_3"].iloc[0]))

    def test_feature_dataframe_formation(self):

        ts_1 = TimeSeries(self.ser_1)
        ts_2 = TimeSeries(self.ser_2)
        ts_3 = TimeSeries(self.ser_3)
        ts_df = TimeSeriesDataframe([ts_1, ts_2, ts_3])

        fd = FeatureDataframe(ts_df, number_of_lags=2)
        feature_df = fd.get_feature_dataframe_not_including_time_t()

        feature_df = fd.get_feature_dataframe_not_including_time_t()
        feature_interpolated_df = fd.get_feature_dataframe_not_including_time_t_interpolated()

        feature_interpolated_no_na_df = fd.get_feature_dataframe_not_including_time_t_no_nan()

        self.assertTrue(len(feature_interpolated_no_na_df) > 0)
        self.assertTrue(len(feature_df) > 0)
        self.assertFalse(feature_interpolated_no_na_df.isnull().values.any())


    def test_develop_training_dataframe(self):

        feature_dataset_names = ["weather", "natural_gas_price_daily"]
        target_dataset_name = "natural_gas_price_daily"
        df = develop_complete_dataframe(feature_dataset_names, target_dataset_name)

        henry_hub_price_time_1 = df["Henry Hub Natural Gas Spot Price Dollars per Million Btu"].iloc[1]
        henry_hub_price_lag_1_time_2 = df["Henry Hub Natural Gas Spot Price Dollars per Million Btu_lag_1"].iloc[2]
        self.assertEqual(henry_hub_price_lag_1_time_2, henry_hub_price_time_1)

        self.assertTrue(len(df) > 1000)
        self.assertTrue(len(df) < 10000)

        self.assertTrue("Target_Diff_Value" in df.columns)
        self.assertTrue("Target_Diff_Sign" in df.columns)

        number_of_negative = len(df[df["Target_Diff_Sign"] == "Positive"])
        number_of_positive = len(df[df["Target_Diff_Sign"] == "Negative"])
        n = len(df)

        self.assertTrue((number_of_negative + number_of_positive == n))
        self.assertTrue(number_of_positive > 0.4 * n)
        self.assertTrue(number_of_negative > 0.4 * n)


    def test_logistic_regression_classification(self):


        run_classification_model(method="logistic_regression")


    def test_neural_network_classification(self):
        run_classification_model(method="neural_network")



if __name__ == '__main__':
    unittest.main()