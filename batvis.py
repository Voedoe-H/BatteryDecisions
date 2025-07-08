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
    test_df = pd.concat(test_segments).reset_index(drop=True)

    X_train = train_df.drop(columns=["RUL", "Cycle_Index"])
    y_train = train_df["RUL"]
    X_test = test_df.drop(columns=["RUL", "Cycle_Index"])
    y_test = test_df["RUL"]

    # params are result of the grid search from the gradient_boosting_approach_optimization function
    params = {
        "n_estimators": 500,
        "max_depth": 3,
        "min_samples_split": 6,
        "learning_rate": 0.03,
        "random_state": 69
    }

    lower_model = ensemble.GradientBoostingRegressor(loss="quantile", alpha=0.05,**params)
    median_model = ensemble.GradientBoostingRegressor(loss="quantile",alpha=0.5, **params)
    upper_model = ensemble.GradientBoostingRegressor(loss="quantile", alpha=0.95,**params)

    lower_model.fit(X_train,y_train)
    median_model.fit(X_train,y_train)
    upper_model.fit(X_train,y_train)

    #reg = ensemble.GradientBoostingRegressor(**params)
    #reg.fit(X_train, y_train)

    #y_pred = reg.predict(X_test)

    #r2 = r2_score(y_test, y_pred)
    #mse = mean_squared_error(y_test, y_pred)
    #mae = mean_absolute_error(y_test, y_pred)

    #print(f"R2 Score: {r2:.4f}")
    #print(f"Mean Squared Error: {mse:.2f}")
    #print(f"Mean Absolute Error: {mae:.2f}")

    #plt.figure(figsize=(12, 5))
    #plt.plot(y_test.values, label='True RUL', alpha=0.7)
    #plt.plot(y_pred, label='Predicted RUL', alpha=0.7)
    #plt.title("Predicted vs. True RUL (Gradient Boosting)")
    #plt.xlabel("Sample Index")
    #plt.ylabel("RUL")
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()

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

    # Optional: print error metrics
    print("MAE:", mean_absolute_error(y_test, median_pred))

def ci_gradient_optimization():
    train_segments, test_segments = data_set_loading("./Battery_RUL.csv")

    train_df = pd.concat(train_segments).reset_index(drop=True)
    test_df = pd.concat(test_segments).reset_index(drop=True)

    X_train = train_df.drop(columns=["RUL", "Cycle_Index"])
    y_train = train_df["RUL"]
    X_test = test_df.drop(columns=["RUL", "Cycle_Index"])
    y_test = test_df["RUL"]

    param_grid = {
        "n_estimators" : [100,200,300,400,500,600,700,800,900,1000],
        "max_depth" : [3,4,5],
        "min_samples_split" : [4,5,6],
        "learning_rate" : [0.01,0.02,0.03],
    }
    
    reg = ensemble.GradientBoostingRegressor(random_state=69,loss="quantile",alpha=0.5)

    grid_search = GridSearchCV(
        estimator=reg,
        param_grid=param_grid,
        cv= 5,
        scoring='r2',
        verbose=2,
        n_jobs=4    
        )

    grid_search.fit(X_train,y_train)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best r2: {grid_search.best_score_}")


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

ci_gradient_optimization()
#gradient_boosting_approach()
#gradient_boosting_approach()
#random_forest_regression()