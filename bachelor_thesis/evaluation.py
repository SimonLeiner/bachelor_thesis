"""
Name: evaluation.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 15.12.21
Description: OOS R^2 Metric definition
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error


def r_sqaured_oos_sklearn(y_true: list, y_pred: list) -> float:
    """

    This function computes the OOS R^2 metric for the hyper parameter sleection

    :param y_true: np.array : the predictions from the model
    :param y_pred: np.array : the true test values
    :return: float : r^ Out of Sample
    """
    # convert the predictions from  a numpy array to a pandas series
    y_pred = pd.Series(y_pred, index=y_true.index)

    # get the histroical mean model predictions
    predictions_hm_model = pd.Series(0, index=y_true.index)

    # the mean squared error of the model
    mse_model = mean_squared_error(y_true=y_true, y_pred=y_pred)

    # the mean squared error of the simple historical mean model (HM)
    mse_benchmark_model = mean_squared_error(y_true=y_true, y_pred=predictions_hm_model)

    # if somethings goes wrong
    if mse_benchmark_model == 0:
        print("Warning : R^2 OOS is arbitrarily set to 0.")  # noqa: T201
        r_squared = 0

    else:
        # calculate the r^2 OOS by Campell and Thompson
        r_squared = 1 - mse_model / mse_benchmark_model

    return r_squared


def r_sqaured_oos_interpretation(
    y_true: list,
    y_pred: list,
    hm_model: callable,
) -> float:
    """

    This function calculate sthe r^2 perfromance metric by campell and Thompson (2008)

    :param y_true: np.array : the predictions from the model
    :param y_pred: np.array : the true test values
    :param hm_model: function : the comparison model
    :return: float : r^ Out of Sample
    """
    # get the histroical mean model predictions
    predictions_hm_model = hm_model(y_true)

    # subset the y_true set and only get the OOS testing values
    y_true = y_true[y_pred.index]

    # the mean squared error of the model
    mse_model = mean_squared_error(y_true=y_true, y_pred=y_pred)

    # the mean squared error of the simple historical mean model (HM)
    mse_benchmark_model = mean_squared_error(y_true=y_true, y_pred=predictions_hm_model)

    # if somethings goes wrong
    if mse_benchmark_model == 0:
        print("Warning : R^2 OOS is arbitrarily set to 0.")  # noqa: T201
        r_squared = 0

    else:
        # calculate the r^2 OOS by Campell and Thompson
        r_squared = 1 - mse_model / mse_benchmark_model

    if len(y_pred) != len(predictions_hm_model):
        print("Something went wrong.")  # noqa: T201

    # investigating how the forecast gains accrued over time
    squared_pred_error_model = (y_true - y_pred) ** 2
    squared_pred_error_hm = (y_true - predictions_hm_model) ** 2

    # cumulative sum
    squared_pred_error_model_cum = squared_pred_error_model.cumsum()
    squared_pred_error_hm_cum = squared_pred_error_hm.cumsum()

    # plotting
    sns.set(rc={"figure.figsize": (8.6074, 3.17), "figure.dpi": 227}, style="ticks")
    sns.lineplot(
        x=y_true.index,
        y=squared_pred_error_hm_cum - squared_pred_error_model_cum,
    )
    plt.ylabel("Cumulative Difference in Squared Prediction Errors")
    plt.savefig(
        "/Users/simonleiner/KIT/Bachelorarbeit/Latex/Thimme/pictures/R_squared_evolution_lasso_0.png",
        bbox_inches="tight",
    )
    plt.show()

    return r_squared, predictions_hm_model, y_true
