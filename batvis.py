import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import List, Tuple
import functools
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import Normal,LogNormal
from ngboost.scores import CRPS
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

def log_function_call(func):
    """
        Logging function
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        print(f"Calling {func.__name__} with args={args} kwargs={kwargs}")
        result = func(*args,**kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_function_call
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

@log_function_call
def tree_regression_approach():
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

@log_function_call
def gradient_boosting_approach():
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")

    train_df = pd.concat(train_segments).reset_index(drop=True)
    test_df = pd.concat(test_segments[:2]).reset_index(drop=True)

    X_train = train_df.drop(columns=["RUL", "Cycle_Index"])
    y_train = train_df["RUL"]
    X_test = test_df.drop(columns=["RUL", "Cycle_Index"])
    y_test = test_df["RUL"]

    # params are result of the grid search from the gradient_boosting_approach_optimization function
    params_median = {
        "n_estimators": 616,
        "max_depth": 7,
        "subsample" : 0.7,
        "min_samples_split": 2,
        "learning_rate": 0.01,
        "random_state": 69
    }

    params_lower = {
        "n_estimators": 764,
        "max_depth": 10,
        "subsample" : 0.7,
        "min_samples_split": 8,
        "learning_rate": 0.29999999999999993,
        "random_state": 69
    }

    params_upper = {
        "n_estimators": 1000,
        "max_depth": 10,
        "subsample" :0.7,
        "min_samples_split": 20,
        "learning_rate": 0.2684362065141949,
        "random_state": 69
    }

    lower_model = ensemble.GradientBoostingRegressor(loss="quantile", alpha=0.01,**params_lower)
    median_model = ensemble.GradientBoostingRegressor(loss="quantile",alpha=0.5, **params_median)
    upper_model = ensemble.GradientBoostingRegressor(loss="quantile", alpha=0.99,**params_upper)

    lower_model.fit(X_train,y_train)
    median_model.fit(X_train,y_train)
    upper_model.fit(X_train,y_train)

    lower_pred = lower_model.predict(X_test)
    median_pred = median_model.predict(X_test)
    upper_pred = upper_model.predict(X_test)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='True RUL', color='black', alpha=0.6)
    plt.plot(median_pred, label='Predicted RUL (median)', color='blue')
    plt.fill_between(range(len(median_pred)), lower_pred, upper_pred, color='blue', alpha=0.2, label='90% CI')
    plt.title("Gradient Boosting: RUL Prediction with Confidence Intervals")
    plt.xlabel("Sample Index")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("MAE:", mean_absolute_error(y_test, median_pred))
    picp = np.mean((lower_pred <= y_test) & (y_test <= upper_pred))
    print(f"PICP: {picp:.2f}")

def ci_gradient_optimization():
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")

    train_df = pd.concat(train_segments).reset_index(drop=True)
    test_df = pd.concat(test_segments).reset_index(drop=True)

    X_train = train_df.drop(columns=["RUL", "Cycle_Index"])
    y_train = train_df["RUL"]
    X_test = test_df.drop(columns=["RUL", "Cycle_Index"])
    y_test = test_df["RUL"]

    searchspace = {
        "n_estimators" : (100,1000),
        "max_depth" : (2,10),
        "learning_rate": (0.01, 0.3, 'log-uniform'),
        "subsample": (0.7, 1.0),
        "min_samples_split": (2, 20)
    }
    
    opt = BayesSearchCV(
        estimator=ensemble.GradientBoostingRegressor(loss='quantile', alpha=0.01),
        search_spaces=searchspace,
        n_iter=50,
        scoring='neg_mean_absolute_error',
        cv=3,
        n_jobs=-1,
        random_state=69
    )

    opt.fit(X_train, y_train)
    print("Best Params:", opt.best_params_)
    print("Best Score:", -opt.best_score_)

def ng_boost_test():
    EPSILON = 1.0

    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")

    train_df = pd.concat(train_segments).reset_index(drop=True)
    test_df = pd.concat(test_segments[:2]).reset_index(drop=True)

    X_train = train_df.drop(columns=["RUL", "Cycle_Index"])
    y_train = train_df["RUL"] + EPSILON
    X_test = test_df.drop(columns=["RUL", "Cycle_Index"])
    y_test = test_df["RUL"] + EPSILON

    model = NGBRegressor(Dist=Normal, Score=CRPS, verbose=True)
    model.fit(X_train, y_train)

    preds = model.pred_dist(X_test)

    y_pred_median = preds.loc - EPSILON

    lower = preds.ppf(0.05) - EPSILON
    upper = preds.ppf(0.95) - EPSILON

    mae = mean_absolute_error(y_test - EPSILON, y_pred_median)
    picp = ((y_test - EPSILON >= lower) & (y_test - EPSILON <= upper)).mean()

    print(f"MAE: {mae:.4f}")
    print(f"PICP (90% CI): {picp:.4f}")

    plt.figure(figsize=(16, 6))
    plt.plot(y_test.values - EPSILON, label='True RUL', color='black')
    plt.plot(y_pred_median, label='Predicted RUL (median)', color='blue')
    plt.fill_between(range(len(y_pred_median)), lower, upper, color='blue', alpha=0.2, label='90% CI')
    plt.title("NGBoost: RUL Prediction with 90% Confidence Interval")
    plt.xlabel("Sample Index")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


@log_function_call
def gradient_boosting_approach_optimization():
    
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")

    train_df = pd.concat(train_segments).reset_index(drop=True)
    test_df = pd.concat(test_segments).reset_index(drop=True)

    X_train = train_df.drop(columns=["RUL", "Cycle_Index"])
    y_train = train_df["RUL"]
    X_test = test_df.drop(columns=["RUL", "Cycle_Index"])
    y_test = test_df["RUL"]

    param_grid = {
        "n_estimators" : [100,300,500],
        "max_depth" : [3,4,5],
        "min_samples_split" : [4,5,6],
        "learning_rate" : [0.01,0.02,0.03],
    }

    reg = ensemble.GradientBoostingRegressor(random_state=69)

    grid_search = GridSearchCV(
        estimator=reg,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',  #'r2'
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    print("Best parameters found:", grid_search.best_params_)
    print("Best CV MSE:", -grid_search.best_score_)

@log_function_call
def random_forest_regression():
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")

    train_df = pd.concat(train_segments).reset_index(drop=True)
    test_df = pd.concat(test_segments).reset_index(drop=True)

    X_train = train_df.drop(columns=["RUL","Cycle_Index"])
    y_train = train_df["RUL"]
    X_test = test_df.drop(columns=["RUL","Cycle_Index"])
    y_test = test_df["RUL"]

    reg = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5 ,random_state=69)
    
    reg.fit(X_train,y_train)
    
    y_pred = reg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values, label='True RUL', alpha=0.7)
    plt.plot(y_pred, label='Predicted RUL', alpha=0.7)
    plt.title("Random Forest: Predicted vs. True RUL")
    plt.xlabel("Sample Index")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def quick_data_analysis():
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")
    all_segments = train_segments + test_segments
    rul_curves = get_fixed_length_rul_curves(all_segments, n_points=100)
    print("Shape of RUL curves array:", rul_curves.shape)
    scaler = StandardScaler()
    rul_curves_scaled = scaler.fit_transform(rul_curves)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(rul_curves_scaled)

    plt.figure(figsize=(10,6))
    for cluster_id in np.unique(labels):
        cluster_curves = rul_curves[labels == cluster_id]
        mean_curve = cluster_curves.mean(axis=0)
        plt.plot(mean_curve, label=f'Cluster {cluster_id} (n={len(cluster_curves)})')
    plt.xlabel("Normalized Cycle Index")
    plt.ylabel("RUL")
    plt.title("Mean Battery Degradation Curves by Cluster")
    plt.legend()
    plt.grid(True)
    plt.show()

def get_fixed_length_rul_curves(segments, n_points=100):
    curves = []
    for seg in segments:
        x = seg["Cycle_Index"].values
        y = seg["RUL"].values
        
        x_norm = (x - x.min()) / (x.max() - x.min())
        
        f = interp1d(x_norm, y, kind='linear')
        
        x_sampled = np.linspace(0, 1, n_points)
        
        y_sampled = f(x_sampled)
        curves.append(y_sampled)
    return np.vstack(curves)



#ci_gradient_optimization()
#ng_boost_test()
#quick_data_analysis()
gradient_boosting_approach()
#random_forest_regression()