import numpy as np
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, pacf
import matplotlib.pyplot as plt

def data_set_loading(path: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
        Loading function
    """
    df = pd.read_csv(path)

    startindices = df.index[df["Cycle_Index"] == 1.0].tolist()

    battery_segments = []
    for i in range(len(startindices)):
        start = startindices[i]
        end = startindices[i + 1] if i + 1 < len(startindices) else len(df)
        battery_segments.append(df.iloc[start:end].reset_index(drop=True))

    num_batteries = len(battery_segments)
    num_train = int(num_batteries * 0.8)

    train_segments = battery_segments[:num_train]
    test_segments = battery_segments[num_train:]

    return train_segments, test_segments

def series_correlation_analysis_singular():
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")

    print(train_segments)
    print(len(train_segments))
    print(len(test_segments))
    print(len(train_segments[0]))
    
    df = train_segments[0]
    columns_to_plot = df.columns.drop(["Cycle_Index"])

    lags = 40
    n_cols = 3
    n_rows = int((len(columns_to_plot) + n_cols - 1) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        try:
            plot_pacf(df[col], lags=lags, ax=axes[i], title=f"PACF: {col}", zero=False)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error\n{str(e)}", ha='center', va='center')
            axes[i].set_title(f"{col} (error)")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def series_correlation_analysis_average():
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")
    lifetimes = train_segments+test_segments 
    features = ['Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)',
                'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)',
                'Charging time (s)', 'RUL']

    max_lag = 40

    pacf_all = {feature: [] for feature in features}

    for df in lifetimes:
        for feature in features:
            series = df[feature].dropna()
            pacf_vals = pacf(series, nlags=max_lag, method='ols')
            pacf_all[feature].append(pacf_vals)

    mean_pacf = {feature: np.mean(pacf_all[feature], axis=0) for feature in features}
    median_pacf = {feature: np.median(pacf_all[feature], axis=0) for feature in features}
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
    axs = axs.flatten()

    for i, feature in enumerate(features):
        axs[i].bar(range(max_lag + 1), mean_pacf[feature])
        axs[i].set_title(f"Mean PACF: {feature}")
        axs[i].set_ylim([-1, 1])

    plt.tight_layout()
    plt.show()

series_correlation_analysis_average()