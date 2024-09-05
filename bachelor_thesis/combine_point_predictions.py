"""
Name: combine_point_predictions.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 11.01.22
Description: Comebine the point predictions from different runs
"""

import pandas as pd


def combine_predictions(  # noqa: PLR0913
    predictions: list,
    y_true: pd.Series,
    min_train_size: int = 160,
    pct_purge: float = 0.00,
    rolling_size: int = 1,
    estimation_horizon: int = 1,
) -> pd.Series:
    """

    This function combines the cross validated prediction into 1 continous pd.Series.

    :param predictions: array : array containing the predictions from the model
    :param y_true: pd.Series : true y values
    :param min_train_size: integer :  minimum datapoints of the training dataset
    :param pct_purge: float : percentage of datapoints to purge between training and testing
    :param rolling_size: integer : size of the rolling window
    :param estimation_horizon: integer : estimation horizon
    :return: pd.Series : predictions from the model
    """
    # when does the first testing set start
    index_test_starts = min_train_size + int(y_true.shape[0] * pct_purge)

    # subset the true values
    corresponding_y_true_vals = y_true[index_test_starts:]

    # empty pd.Series to save values if we watn to aggreagate all point predictions into one padans series
    series_vals = pd.Series()

    # loop over all cv predicitons
    for _, prediction in enumerate(predictions):
        # predictions with proper time index
        prediction = pd.Series(  # noqa: PLW2901
            prediction,
            index=corresponding_y_true_vals.index[:estimation_horizon],
        )

        # append the value
        series_vals = series_vals.append(prediction)

        # always roll forward : delte the rolling_size first datapoints
        corresponding_y_true_vals = corresponding_y_true_vals[rolling_size:]

    return series_vals
