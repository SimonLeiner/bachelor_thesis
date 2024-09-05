"""
Name: hm_models.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 21.01.22
Description: Differnet benchmark models for computation of the R^2 metric
"""

import pandas as pd


def hm_0_model(y_true: pd.Series, min_train_size: int) -> pd.Series:
    """

    This function computes the simple historical mean model as a constant value 0.

    :param y_true: pd.Series : the true y values
    :param min_train_size: integer : the minimum training size of the ML model
    :return: pd.Series : predictions of the historical average model

    """
    # compute the simple average model from the full y_true data
    predictions_hm_model = pd.Series(0, index=y_true.index)

    # exclude first 160 data points
    return predictions_hm_model[min_train_size:]


def rolling_12_hm_model(y_true: pd.Series, min_train_size: int) -> pd.Series:
    """

    This function computes the simple historical mean model with a rolling window of 12 dp =  1 year.

    :param y_true: pd.Series : the true y values
    :param min_train_size: integer : the minimum training size of the ML model
    :return: pd.Series : predictions of the historical average model

    """
    # compute the simple average model from the full y_true data
    predictions_hm_model = y_true.rolling(window=12).mean()

    # shift the model: real time investor : CRITICAL
    predictions_hm_model = predictions_hm_model.shift(1)

    # exclude first 160 data points
    return predictions_hm_model[min_train_size:]
