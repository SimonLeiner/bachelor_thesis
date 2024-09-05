"""
Name: create_csv_files.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 04.02.22
Description:
"""

import calculate_returns
import pandas as pd


def execute_function() -> None:
    # Note: returns from the 1984-03-01 up to 2017-08-01 and returns are always realized at the end of each month

    # wheter to forecast or explain the equity risk premia
    forecast = "innovation"

    # get the sp500 data
    df_sp500 = pd.read_csv(
        "/Users/simonleiner/PycharmProjects/bachelorarbeit/csv_files/MULTPL-SP500_REAL_PRICE_MONTH.csv",
        header=0,
        sep=",",
        thousands=None,
    )

    # adjust the date column
    df_sp500["Date"] = pd.to_datetime(df_sp500["Date"], errors="raise")

    # set the date column as the index
    df_sp500 = df_sp500.set_index("Date")

    # data is in reverse !! 2017 at the top so sort!!!
    df_sp500 = df_sp500.sort_index(ascending=True)

    # calculate the returns: loose first datapoint
    df_sp500 = calculate_returns.calculate_log_returns(df_sp500, "Value")

    # get the 3 month us treasury bill rate in %
    df_rate = pd.read_csv(
        "/Users/simonleiner/PycharmProjects/bachelorarbeit/csv_files/F-F_Research_Data_Factors.CSV",
        header=0,
        sep=",",
        thousands=None,
    )

    # only get the relevant data
    df_rate = df_rate.drop(["date", "Mkt-RF", "SMB", "HML"], axis=1)

    # set the index accordingly
    df_rate = df_rate.set_index(df_sp500.index)

    # can't be in decimlas, because we would by 0.5 have a risk free rate of 50%
    # https://fred.stlouisfed.org/series/DGS1MO -> must divide by 100
    df_rate = df_rate / 100

    # calculate the excess returns
    df_sp500["exc_return"] = df_sp500["return"] - df_rate["RF"]

    # delete added columns
    df_sp500 = df_sp500.drop(["Value", "return"], axis=1)

    # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Note: text data from the 1984-01-01 up to 2017-06-01 and text data is alwys included at the beginning of the month

    # get the text data
    df_text = pd.read_csv(
        "/Users/simonleiner/PycharmProjects/bachelorarbeit/csv_files/Monthly_Topic_Attention_Theta.csv",
        header=0,
        sep=",",
        thousands=None,
    )

    # adjust the date column
    df_text["date"] = pd.to_datetime(df_text["date"], errors="raise")

    # set the date column as the index
    df_text = df_text.set_index("date")

    # forecasting, naturally matched : text data from the 2017-05-01 match with returns from the 2017-07-01
    if forecast is True:
        pass

    # explaining, create artificail return on the 1983-02-01
    elif forecast is False:
        df_sp500 = df_sp500.shift(1)

        df_sp500.index = df_sp500.index - pd.offsets.MonthBegin()

        df_sp500 = df_sp500.fillna(0)

    # explaining, create artificail return on the 1983-02-01
    elif forecast == "forecast_news":
        df_sp500 = df_sp500.shift(2)

        df_sp500.index = df_sp500.index - pd.offsets.MonthBegin()
        df_sp500.index = df_sp500.index - pd.offsets.MonthBegin()

        df_sp500 = df_sp500.fillna(0)

    elif forecast == "innovation":
        df_text = df_text - df_text.shift(1)

        df_text = df_text.fillna(0)

    else:
        pass

    # set the index accordingly: we take the index of teh asset prices!!
    df_text = df_text.set_index(df_sp500.index)

    # combine the 2 dataframes: Take the datetime index from the text data
    df_combined = pd.concat([df_text, df_sp500], axis=1)

    # convert the pandas DataFrame into a csv File
    df_combined.to_csv(f"combined_data_forecast_{forecast}.csv")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    execute_function()
