import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import List, Tuple
import functools

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        print(f"Calling {func.__name__} with args={args} kwargs={kwargs}")
        result = func(*args,**kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_function_call
def data_set_loading(path: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
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

train_segments, test_segments = data_set_loading("./Battery_RUL.csv")

train_df = pd.concat(train_segments).reset_index(drop=True)
test_df = pd.concat(test_segments).reset_index(drop=True)

X_train = train_df.drop(columns=["RUL","Cycle_Index"])
y_train = train_df["RUL"]
X_test = test_df.drop(columns=["RUL","Cycle_Index"])
y_test = test_df["RUL"]

pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_regression, k=5)),
    ('regressor', DecisionTreeRegressor(random_state=69))
])

pipeline.fit(X_train,y_train)

selector = pipeline.named_steps['feature_selection']
mask = selector.get_support()

selected_feature_names = X_train.columns[mask]

print("Selected features:", selected_feature_names.tolist())

y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Rsq Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label='True RUL', alpha=0.7)
plt.plot(y_pred, label='Predicted RUL', alpha=0.7)
plt.title("Predicted vs. True RUL")
plt.xlabel("Sample Index")
plt.ylabel("RUL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
