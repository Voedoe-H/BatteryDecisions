import numpy as np
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, pacf, ccf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

def cross_correlation(x, y, max_lag):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    result = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(x[-lag:], y[:lag])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]
        result.append(corr)
    return np.array(result)

def series_cross_correlation():
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")
    df = train_segments[0]
    x = df['Charging time (s)'].dropna().values
    y = df['Min. Voltage Charg. (V)'].dropna().values

    cc_vals = cross_correlation(x, y, max_lag=20)

    plt.plot(range(-20, 21), cc_vals)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Cross-Correlation: Charging Time vs Min. Voltage Charg.')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.show()

def compute_global_normalization_params(segments: List[pd.DataFrame], features: List[str]):
    concat = pd.concat([df[features] for df in segments], axis=0)
    mean = concat.mean()
    std = concat.std()
    return mean, std

class BatteryDataset(Dataset):
    def __init__(self, 
                 segments: List[pd.DataFrame], 
                 sequence_length: int, 
                 features: List[str], 
                 mean: pd.Series, 
                 std: pd.Series):
        
        self.samples = []
        self.targets = []
        self.sequence_length = sequence_length

        for segment in segments:
            segment = segment.copy()
            segment[features] = (segment[features] - mean[features]) / std[features]

            data = segment[features].values
            rul = segment["RUL"].values

            for i in range(len(segment) - sequence_length):
                x = data[i:i+sequence_length]
                y = rul[i+sequence_length]
                self.samples.append(x)
                self.targets.append(y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()

def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss += criterion(val_outputs, y_val).item() * x_val.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output.squeeze(), y)
            total_loss += loss.item() * x.size(0)

            preds.append(output.cpu().squeeze().numpy())
            targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    return avg_loss, preds, targets


features = ['Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)',
            'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)',
            'Charging time (s)']

train_segments, test_segments = data_set_loading("./Battery_RUL.csv")

mean, std = compute_global_normalization_params(train_segments, features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 30
train_dataset = BatteryDataset(train_segments, sequence_length, features, mean, std)
test_dataset = BatteryDataset(test_segments, sequence_length, features, mean, std)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_dim = len(features)
model = LSTMModel(input_dim=input_dim)

trained_model = train_model(model, train_loader, val_loader, num_epochs=65, lr=1e-3)

test_loss, predicted_rul, actual_rul = evaluate_model(model, val_loader, device)
print(f"Test MSE: {test_loss:.4f}")

actual_rul = np.array(actual_rul)
predicted_rul = np.array(predicted_rul)

r2 = r2_score(y_true=actual_rul,y_pred=predicted_rul)
mse = mean_squared_error(actual_rul,predicted_rul)
mae = mean_absolute_error(actual_rul,predicted_rul)

print(f"r2: {r2:4f}")
print(f"mse: {mse:4f}")
print(f"mae: {mae:4f}")

plt.figure(figsize=(12, 6))
plt.plot(actual_rul, label="True RUL", linewidth=2)
plt.plot(predicted_rul, label="Predicted RUL", linestyle='--', linewidth=2, alpha=0.8)
plt.title("Predicted vs. True RUL")
plt.xlabel("Sample Index")
plt.ylabel("Remaining Useful Life (RUL)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(actual_rul, predicted_rul, alpha=0.6)
plt.plot([actual_rul.min(), actual_rul.max()],
         [actual_rul.min(), actual_rul.max()], 'r--', label="Ideal")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("True vs. Predicted RUL (Scatter)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()