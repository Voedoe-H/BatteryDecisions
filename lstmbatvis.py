import numpy as np
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

def series_correlation_analysis():
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")
    
    print(train_segments)
    print(len(train_segments))
    print(len(test_segments))
    print(len(train_segments[0]))

    