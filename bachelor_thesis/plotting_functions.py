"""
Name: plotting_functions.py in Project: bachelorarbeit
Author: Simon Leiner
Date: 20.10.21
Description: Some useful plotting functions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from evaluation import r_sqaured_oos_interpretation


def corrheatmap(df: pd.DataFrame, ycol: str, k: int) -> None:
    """

    This function computes a pretty correlation heatmap.

    :param df: pd.DataFrame : Dataframe
    :param ycol: string :  name of the y-column
    :param k: integer : number of columns to show in matrix
    :return: None
    """
    # set the title
    plt.title("Correlation Heatmap:\n")

    # get the k largest columns by pearson coeficient and select the index thereof
    cols = df.corr().abs().nlargest(k, ycol)[ycol].index

    # get the correlation coeficcient of the selected columns : Transpose the Array containing the values
    cm = np.corrcoef(df[cols].to_numpy().T)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(cm, dtype=bool))

    # set the font scalte
    sns.set(font_scale=1.25)

    # plot the heatmap
    sns.heatmap(
        cm,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        yticklabels=cols.values,
        xticklabels=cols.values,
        mask=mask,
    )

    plt.show()


def plot_predictions_cv(
    y_true: pd.Series,
    y_pred: pd.Series,
    hm_model: callable,
) -> None:
    """

    This function plots the predictions and true values of the trained model.

    :param y_true: pd.Series : true y values
    :param y_pred: pd.Series : predicted y values from the main model
    :param hm_model: function : the comparison model
    :return: None
    """
    # evaluation OOS
    r_squred, predictions_hm_model, corresponding_y_true_vals = (
        r_sqaured_oos_interpretation(y_true=y_true, y_pred=y_pred, hm_model=hm_model)
    )
    print(f"R^2 OOS: {round(r_squred,4)*100} %")  # noqa: T201
    print("-" * 10)  # noqa: T201

    # plotting
    sns.set(rc={"figure.figsize": (8.6074, 3.17), "figure.dpi": 227}, style="ticks")
    sns.lineplot(
        x=corresponding_y_true_vals.index,
        y=corresponding_y_true_vals,
        label="Realized Values",
    )
    sns.lineplot(x=y_pred.index, y=y_pred, label="Predicted ML Values")
    sns.lineplot(
        x=predictions_hm_model.index,
        y=predictions_hm_model,
        label="Predicted HM Values",
    )
    plt.ylabel("S&P500 excess returns")
    plt.legend(loc="upper right")
    plt.show()

    # for density plotting
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    df1["return"] = corresponding_y_true_vals
    df1["Series"] = "true values"

    df2["return"] = y_pred
    df2["Series"] = "predicted ML values"

    df_plotting = pd.concat([df1, df2], ignore_index=True)

    # plotting
    sns.set(rc={"figure.figsize": (8.6074, 3.17), "figure.dpi": 227}, style="ticks")
    sns.displot(
        data=df_plotting,
        x="return",
        hue="Series",
        fill=True,
        stat="density",
        kde=True,
        legend=False,
    )
    plt.xlabel("S&P500 excess returns")
    plt.legend(["Predicted ML Values", "Realized Values"], loc="best")
    plt.show()

    print(  # noqa: T201
        f"Average: | true values: {round(corresponding_y_true_vals.mean(),4)} | predicted ML values: {round(y_pred.mean(),4)} | predicted HM model: {round(predictions_hm_model.mean(),4)}",
    )
    print("-" * 10)  # noqa: T201
