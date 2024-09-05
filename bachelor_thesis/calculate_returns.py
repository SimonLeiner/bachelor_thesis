"""
Name: calculate_returns.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 15.01.22
Description: calculate returns from prices and reverse
"""

import numpy as np
import pandas as pd


def calculate_log_returns(df: pd.DataFrame, pricecolname: str) -> pd.DataFrame:
    """

    This function computes the log returns

    :param df: pd.Dataframe : Dataframe with the price information
    :param pricecolname: string : name of the price column
    :return: pd.Dataframe : Dataframe with added column
    """
    # calculate the log returns : log, because addative over time
    df["return"] = np.log(df[pricecolname] / df[pricecolname].shift(1))

    # drop the na row
    return df.dropna(axis=0)


def calculate_price_from_log_returns(
    initial_price: float, df: pd.DataFrame, returncolname: str
) -> pd.DataFrame:
    """

    This function calculates the price from an inital investment and returns

    :param initial_price: float : first starting price
    :param df: pd.Dataframe : Dataframe with the log return information
    :param returncolname: string : name of the return column
    :return: pd.Dataframe : Dataframe with the added column
    """
    # list to save values
    prices_list = []

    # iterate over rows
    for _, row in df.iterrows():
        price = initial_price * np.exp(row[returncolname])

        initial_price = price

        prices_list.append(price)

    df["price"] = prices_list

    return df
