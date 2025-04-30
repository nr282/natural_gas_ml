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
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from weather import get_all_temperature_data
from sklearn.neural_network import MLPClassifier


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

        return self.feature_dataframe_not_including_time_t.interpolate().ffill().dropna()

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

def get_names_for_time_column():

    return ["Date", "Time"]

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
            #Don't want to add time or date column
            if column.title() in get_names_for_time_column():
                pass
            else:
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
    feature_dataframe = FeatureDataframe(time_series_df, number_of_lags=3)
    feature_dataframe = feature_dataframe.get_feature_dataframe_not_including_time_t_interpolated()

    #Grab the target series.
    target_dataset_series = datasets.get(target_time_series)
    target_dataset_series_dict = dict([(ts.name, ts) for ts in target_dataset_series])
    target_series_name = "Henry Hub Natural Gas Spot Price Dollars per Million Btu"
    target_dataset_series = target_dataset_series_dict.get(target_series_name)

    target_time_series = TargetTimeSeries(target_time_series,
                                          target_dataset_series)




    training_df = merge_feature_target(feature_dataframe, target_time_series)
    training_df["Target_Diff_Value"] = training_df[target_series_name].diff().bfill()
    training_df["Target_Diff_Sign"] = training_df["Target_Diff_Value"].apply(lambda x: "Positive" if x > 0 else "Negative")

    return training_df


def split_dataset(df: pd.DataFrame):



    train, validate, test = \
        np.split(df.sample(frac=1, random_state=42),
                 [int(.6 * len(df)), int(.8 * len(df))])




    train_df = pd.DataFrame(train)
    validate_df = pd.DataFrame(validate)
    test_df = pd.DataFrame(test)

    return train_df, validate_df, test_df



def train_model_with_lasso_regression(train_df: pd.DataFrame):
    """
    Train Model with Lasso Regression.


    :return:
    """


    pass


def filter_dataframe(completed_df, target_time_series_name):

    feature_columns = list(filter(lambda x: ("lag" in x.lower() or x.lower() == target_time_series_name.lower()),
                                  list(completed_df.columns)))


    return completed_df[feature_columns]

def train_model_with_logistic_regression(x_train_pd: pd.DataFrame,
                                         y_train_pd: pd.DataFrame,
                                         x_test_pd: pd.DataFrame,
                                         y_test_pd: pd.DataFrame):

    x_train = x_train_pd.to_numpy()
    y_train = y_train_pd.to_numpy()
    x_test = x_test_pd.to_numpy()
    y_test = y_test_pd.to_numpy()

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=100000,
                               class_weight="balanced",
                               solver="newton-cholesky")


    model.fit(x_train, y_train)

    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f}% \n ".format(accuracy * 100))


def train_model_with_neural_network(x_train_pd: pd.DataFrame,
                                         y_train_pd: pd.DataFrame,
                                         x_test_pd: pd.DataFrame,
                                         y_test_pd: pd.DataFrame):


    x_train = x_train_pd.to_numpy()
    y_train = y_train_pd.to_numpy()
    x_test = x_test_pd.to_numpy()
    y_test = y_test_pd.to_numpy()

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-2,
                        hidden_layer_sizes=(5, 5), random_state=1,
                        max_iter=100000)

    clf.fit(x_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Neural Network Accuracy: {:.2f}% \n ".format(accuracy * 100))


def run_classification_model(method = "logistic_regression"):

    #Update weather.
    start = datetime.datetime(1997, 1, 1)
    end = datetime.datetime(2025, 3, 31)
    get_all_temperature_data(start, end)


    feature_dataset_names = ["weather", "natural_gas_price_daily"]
    target_dataset_name = "natural_gas_price_daily"
    target_column_name = "Target_Diff_Sign"
    df = develop_complete_dataframe(feature_dataset_names, target_dataset_name)
    df = filter_dataframe(df, target_column_name)  # May be too brittle to future changes.
    train_df, test_df, validate_df = split_dataset(df)
    train_columns = set(train_df.columns)
    df_columns = set(df.columns)

    x_train_pd = train_df.drop([target_column_name], axis=1)
    y_train_pd = train_df[target_column_name]

    x_test_pd = test_df.drop([target_column_name], axis=1)
    y_test_pd = test_df[target_column_name]

    negative_train = y_train_pd.str.count("Negative").sum()
    positive_train = y_train_pd.str.count("Positive").sum()

    logging.info(f"Percent Negative Days {negative_train / (negative_train + positive_train) * 100}%")
    logging.info(f"Percent Positive Days {positive_train / (negative_train + positive_train) * 100}%")
    logging.info(f"Number of datapoints in train dataframe {len(train_df)}")




    negative_test = y_test_pd.str.count("Negative").sum()
    positive_test = y_test_pd.str.count("Positive").sum()

    logging.info(f"Percent Negative Days {negative_test / (negative_test + positive_test) * 100}%")
    logging.info(f"Percent Positive Days {positive_test / (negative_test + positive_test) * 100}%")
    logging.info(f"Number of datapoints in test dataframe {len(test_df)}")









    if method == "neural_network":
        train_model_with_neural_network(x_train_pd,
                                             y_train_pd,
                                             x_test_pd,
                                             y_test_pd)
    elif method == "logistic_regression":
        train_model_with_logistic_regression(x_train_pd,
                                        y_train_pd,
                                        x_test_pd,
                                        y_test_pd)
    else:
        pass












