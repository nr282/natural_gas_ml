"""
In the following module, it will be critical to build up the natural gas prices into a
daily dataset of prices. The prices will be the target variable for the training.


"""
import os
import pandas as pd


def load_natural_gas_prices():

    natural_gas_price_file_name = "Henry_Hub_Natural_Gas_Spot_Price.csv"
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "natural_gas_price", natural_gas_price_file_name))

    return df