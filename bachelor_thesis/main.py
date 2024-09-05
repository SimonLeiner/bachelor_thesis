"""
Name: main.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 18.10.21
Description: executable file in the project
"""

import time

import hm_models
import hyper_parameter
import interprate_model
import pandas as pd
import seaborn as sns
from combine_point_predictions import combine_predictions
from evaluation import r_sqaured_oos_sklearn
from get_predicitons_score import cross_validate_adjusted
from plotting_functions import plot_predictions_cv
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer
from time_series_split import TimeSeriesSplit_adjusted


def execute_function() -> None:
    """

    This function executes the script

    :return: None

    """
    # colors to use
    # colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']

    # set the plotting theme
    sns.set_theme(style="ticks")  # ,palette=colors)

    # track the time
    starttime = time.perf_counter()

    # hyper parameters tuned
    tune = True

    # forecast or explain setting, True, False, "forecast_news", innovation
    forecast = True

    # percenteage of datapoints to purge between the train and testing set
    pct_purge = 0.0

    # Checked : Training size = 160 dp
    min_train_size = 160

    # Checked : rolling size =  1 dp
    rolling_size = 1

    # Checked : minimum testing size = 1 dp
    min_test_size = 0

    # Checked : estimation horizon = 1 dp
    estimation_horizon = 1

    # perfromance metric must be convertet to a scorer function
    scorer = make_scorer(r_sqaured_oos_sklearn)

    print("-" * 10)  # noqa: T201

    # get the dataframe
    df = pd.read_csv(
        f"/Users/simonleiner/PycharmProjects/bachelorarbeit/csv_files/combined_data_forecast_{forecast}.csv",
        header=0,
    )

    # adjust the date column
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    # set the date column as the index
    df = df.set_index("date")

    # get the data in a machine Learning setting : dependent and independent variables
    X, y = df.drop(["exc_return"], axis=1), df["exc_return"]  # noqa: N806

    # rolling cross validation object
    cv = TimeSeriesSplit_adjusted(
        pct_purge=pct_purge,
        min_test_size=min_test_size,
        min_train_size=min_train_size,
        rolling_size=rolling_size,
        estimation_horizon=estimation_horizon,
    )

    # Note : select the model and create the model with the default hyper params, RandomForestRegressor
    model = Lasso()

    # Note : select the benchmark model for OOS Comparison
    def hm_model(y_true: pd.Series) -> pd.Series:
        return hm_models.hm_0_model(y_true, min_train_size)

    # if the hyper parameters are fixed, but not the case here
    if tune is False:
        # if the hyper params have been tuned before
        if type(model).__name__ == "Lasso":
            param_grid = {"alpha": 0.000005}

        else:
            param_grid = {"max_depth": 100, "n_estimators": 250, "n_jobs": -1}

    else:
        # grid for hyper parameter search
        param_grid = hyper_parameter.get_hyper_tuning(model)

    # cross validation scores
    results = cross_validate_adjusted(
        model,
        X,
        y,
        cv=cv,
        param_grid=param_grid,
        tune=tune,
        return_predictions=True,
        scoring=scorer,
        return_estimator=True,
        n_jobs=1,
    )

    # predictions saved in an array in an array : n_splits arrays in array
    predictions = results["predictions"]

    # get the fitted models
    trained_models = results["estimator"]

    # combine the predictions
    predictions_combined = combine_predictions(
        predictions,
        y,
        min_train_size=min_train_size,
        pct_purge=pct_purge,
        rolling_size=rolling_size,
        estimation_horizon=estimation_horizon,
    )

    # plot all predictions
    plot_predictions_cv(y, predictions_combined, hm_model=hm_model)

    # interprate the Lasso model
    if type(trained_models[0]).__name__ == "Lasso":
        interprate_model.interprate_lasso(trained_models, X, 15)

    # interpretation for random forest
    if type(trained_models[0]).__name__ == "RandomForestRegressor":
        interprate_model.feature_importance_rf(trained_models, X, 15)

    print(f"Script time : {time.perf_counter() - starttime} seconds")  # noqa: T201
    print("-" * 10)  # noqa: T201


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    execute_function()
