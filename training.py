"""
Handles the training data.


========================================================================================================================
DATA MODELLING DIAGRAM

The key element here is that we need to match up:
    1. natural gas prices
    2. weather data

There will be a particular data pipeline used.
    1. Time Series Dataframe.
        - The dataframe contains data of the following
        form:

        Feature 1   Feature 2   Feature 3   Feature 4   Feature 5  Target Price

    2. Feature Dataframe with a particular window.


    f(1,1), f(1,2), ... f(1,N), f(2,1), f(2,2), ... f(2,N), f(3,1), f(3,2), ... f(3,N)      Target Price


The key element is to move from (1) Time Series Dataframe to (2) Feature Dataframe with Particular Window.

In a diagram, our first pass will look something like this.
    Time Series Dataframe ------> Feature Dataframe

In order to allow for the quick incorporation of new data into the dataframe, we will continue to setup the
pipeline to make it more robust and scaleable, at least in the code. Hence, a second pass of the diagram is provided
below:

    Time Series 1 ----

    Time Series 2 ----   Time Series Dataframe ------> Feature Dataframe

    Time Series 3 ----

After one has the Feature Dataframe, one can look to split it to perform the three major divisions that
are necessary: (1) Test, (2) Train, (3) Validate.

========================================================================================================================
MODEL TRAINING DIAGRAM

A model can be developed to aid in the Training phase of the below process.

The model will be denoted by "M".

Create Model M  ------------v
                            v
                            v
Feature Dataframe ----> Training(M) -----> Model Parameters for M, and Model are the outputs.
                                           I can save the Model Parameters for M and the model M to a
                                           s3 database.

Upon the inference stage, we take in the model M and M parameters, and we apply them to incoming
data. In a diagram, this looks like the below.

========================================================================================================================
INFERENCE DIAGRAM


Incoming Data Request

Saved Model M with Model Parameters --->
                                        v
                                        v
                                        v
                                        v
Incoming Data Request -----------------> Inference  ------> Prediction.



========================================================================================================================
MACHINE LEARNING ALGORITHM DESIGN

Models to try will initially include vanilla versions directly from tensorflow for these elements:
    1. LASSO
    2. NEURAL NETWORK
    3. SIMPLE LINEAR REGRESSION
    4.

I also should look to do some research on tensorflow and how we can place our material in there.

Many people I have spoken to state that all we need to do is use a binary classifier.



"""
import os.path

import pandas as pd
from weather import get_all_temperature_data
import tensorflow as tf
import datetime
import os
import logging
from dateutil.parser import parse
from functools import reduce

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def get_global_path():
    return "C:\\Users\\nr282\\PycharmProjects\\natural_gas_deep_learning"

class TimeSeries(object):
    """
    Time Series Represents data that comes in a time series format.

    A time series is represented below:

    Date                Value
    11-01-2025          10
    11-02-2025          10
    11-03-2025          11

    """

    def __init__(self, ts: pd.Series):

        self.series = ts
        self._size = ts.size
        self._start_date = ts.index.min()
        self._end_date = ts.index.max()
        self._name = ts.name
        self._frequency = pd.infer_freq(ts.index)
        self._type_of_data = ts.dtype

    def get_series(self) -> pd.Series:
        return self.series

    @property
    def size(self):
        return self._size

    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    @property
    def name(self):
        return self._name

    @property
    def frequency(self):
        return self._frequency

    @property
    def type_of_data(self):
        return self.type_of_data

    def average(self):
        if self.type_of_data == float or self.type_of_data == int:
            return self.series["Value"].mean()
        else:
            return None


class TimeSeriesDataframe(object):
    """
    The TimeSeriesDataframe object takes in a list of TimeSeries and computes
    the resulting TimeSeriesDataframe.


    """

    def __init__(self, ts_list: list):

        self.ser_list = [ser.get_series() for ser in ts_list]
        self.names = set([ts.name for ts in ts_list])

        df = pd.concat(self.ser_list,
                       join='outer',
                       axis=1)

        self.df = df

    def get_dataframe(self):
        return self.df




class FeatureDataframe(object):
    """
    Feature Dataframe is developed from the TimeSeriesDataframe.

    Feature Dataframe takes in the time series data and computes a
    filtration, which is all of the available data available up
    to a particular time. What this means is that at time T, the feature
    dataframe includes features that are the time series lagged by T-1,
    T-2, up to T-k.

    """


    def __init__(self,
                 ts_df: TimeSeriesDataframe,
                 number_of_lags : int = 1):

        self.ts_df = ts_df
        self.base_feature_names = ts_df.df.columns
        self.number_of_lags = number_of_lags
        self.feature_dataframe_including_time_t = self.__calculate_feature_dataframe_with_lag(ts_df)
        self.feature_dataframe_not_including_time_t = self.__calculate_feature_dataframe_not_including_time_t()
        self.feature_dataframe_not_including_time_t_interpolated = self.__calculate_feature_dataframe_not_including_time_t_interpolated()
        self.feature_dataframe_not_including_time_t_no_nan = self.__calculate_feature_dataframe_not_including_time_t_dropna()

    def __calculate_feature_dataframe_with_lag(self,
                                            ts_df: TimeSeriesDataframe):
        """
        The goal here is to develop a dataframe that represents a set of experiments
        starting with a time start time and ending with an end time.

        The goal is to have all the features lagged up to a particular time.


        :param ts_df:
        :return:
        """

        #NOTE: This could be made more efficient.
        df = ts_df.df
        columns = list(df.columns)
        for column in columns:
            for lag in range(1, self.number_of_lags + 1):
                df[column + "_" + 'lag_' + str(lag)] = df[column].shift(lag)

        return df

    def __calculate_feature_dataframe_not_including_time_t_interpolated(self):

        return self.feature_dataframe_not_including_time_t.interpolate().ffill().bfill()

    def __calculate_feature_dataframe_not_including_time_t_dropna(self):
        return self.feature_dataframe_not_including_time_t.dropna()

    def __calculate_feature_dataframe_not_including_time_t(self):

        return self.feature_dataframe_including_time_t.drop(columns=self.base_feature_names)


    def get_feature_dataframe_including_time_t(self):
        return self.feature_dataframe_including_time_t

    def get_feature_dataframe_not_including_time_t(self):
        return self.feature_dataframe_not_including_time_t


    def get_feature_dataframe_not_including_time_t_interpolated(self):
        return self.feature_dataframe_not_including_time_t_interpolated

    def get_feature_dataframe_not_including_time_t_no_nan(self):
        return self.feature_dataframe_not_including_time_t_no_nan


class TargetTimeSeries(object):
    """
    Target Variable includes Natural Gas Prices.

    """

    def __init__(self, target_variable: str, ser: pd.Series):

        self.target_variable_name = target_variable
        self.target_series = ser


    def get_target_series(self):
        return self.target_series

    def get_target_variable_name(self):
        return self.target_variable_name




def dataset_loader(dataset_name: str):
    """
    Loads dataset with dataset name.

    """

    df = None
    try:
        logging.info("Attempting to load dataset {}".format(dataset_name))
        path = dataset_configuration(dataset_name)

        df = pd.read_csv(str(path))
        logging.info("Successfully loaded dataset {}".format(dataset_name))
    except:
        pass


    return df


def convert_date_to_datetime(df):
    #04/7/2025

    df["Date"] = df["Date"].apply(lambda date_str: parse(date_str))

    return df

def format_dataframe(df: pd.DataFrame, formats):
    """
    Goal here is to take dataframe passed in and manipulate into
    a format as seen below:

    -------------------------------------------------------------

    Date    |   Feature 1       Feature 2       Feature 3
            |
    Index   |
    -------------------------------------------------------------

    The input dataframe may have a specific format.


    :param df:
    :return:
    """

    if not ("Date" in df.columns or "time" in df.columns or "Day" in df.columns):
        raise ValueError("No time dimension found in dataframe")

    df = df.rename(columns={"time": "Date"})
    df = df.rename(columns={"Day": "Date"})

    df = convert_date_to_datetime(df)
    df = df.set_index("Date", drop=False)

    max_occurences_of_dates = int(df["Date"].value_counts().max())
    if max_occurences_of_dates == 1:
        return df
    elif max_occurences_of_dates > 1:
        df_pivot = df.pivot(index='Date',
                            columns=formats.get("feature_names"),
                            values=formats.get("value"))

        new_columns = df_pivot.columns.map(','.join)
        df_pivot.columns = new_columns
        return df_pivot
    else:
        raise



def load_datasets(dataset_names: dict[str]):
    """
    Loads all the datasets corresponding to dataset names.

    :param dataset_names:
    :return:
    """

    datasets = dict()
    for dataset_name in dataset_names:
        df = dataset_loader(dataset_name)

        formats = dataset_features_configuration(dataset_name)

        #The dataframe loaded from the dataframe is a raw dataframe
        #and may require manipulation.

        formatted_df = format_dataframe(df, formats)
        time_series_set = set()
        for column in formatted_df.columns:

            time_series_set.add(TimeSeries(formatted_df[column]))

        datasets[dataset_name] = time_series_set


    return datasets

def dataset_configuration(dataset_name):


    configuration = {"weather" : os.path.join(get_global_path(), "data", "weather", "weather_data.csv"),
                     "natural_gas_price_monthly": os.path.join(get_global_path(), "data", "natural_gas_price", "natural_gas_price_monthly.csv"),
                     "natural_gas_price_daily": os.path.join(get_global_path(), "data", "natural_gas_price",
                                                               "natural_gas_price_daily.csv")
    }

    return configuration.get(dataset_name)

def dataset_features_configuration(dataset_name):


    configuration = {"weather": {"feature_names": ["City", "State"],
                                 "value": "tavg"}}

    return configuration.get(dataset_name)


def merge_feature_target(feature_dataframe: FeatureDataframe,
                         target_time_series: TargetTimeSeries):


    target_time_series_values = target_time_series.get_target_series().get_series()

    df = feature_dataframe.merge(target_time_series_values,
                                 left_index=True,
                                 right_index=True)
    return df

def develop_complete_dataframe(feature_dataset_names: list[str],
                               target_time_series: str):
    """
    Develop a dataframe that incorporates a set of data sources. The data sources are
    pulled from dataset names.

        1. Weather Data
        2. Natural Gas Price.

    """

    #Find datasets corresponding to dataset names.
    #Load the datasets.
    datasets = load_datasets(feature_dataset_names)
    datasets_vals = list(reduce(lambda x, y: x.union(y), datasets.values()))
    time_series_df = TimeSeriesDataframe(datasets_vals)

    #Complete the FeatureDataframe.
    feature_dataframe = FeatureDataframe(time_series_df, number_of_lags=5)
    feature_dataframe = feature_dataframe.get_feature_dataframe_not_including_time_t_interpolated()

    #Grab the target series.
    target_dataset_series = datasets.get(target_time_series)
    target_dataset_series_dict = dict([(ts.name, ts) for ts in target_dataset_series])
    target_series_name = "Henry Hub Natural Gas Spot Price Dollars per Million Btu"
    target_dataset_series = target_dataset_series_dict.get(target_series_name)

    target_time_series = TargetTimeSeries(target_time_series,
                                          target_dataset_series)


    training_df = merge_feature_target(feature_dataframe, target_time_series)

    return training_df











