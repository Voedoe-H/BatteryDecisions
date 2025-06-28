import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def all_trip_loading():
    winter_trips = []

    for file in os.listdir("./CarBat"):
        if file.startswith("TripB"):
            winter_trips.append(file)

    for trip in winter_trips:
        df = pd.read_csv(f"./CarBat/{trip}", encoding='latin1')
        print(df)

def feature_analysis():
    df = pd.read_csv("./CarBat/TripB01.csv",sep=";", encoding='cp1252')
    print(df.columns)
    
    trip_traces_X = []
    trip_traces_Y = []

    for j in range(5):
        dfl = pd.read_csv(f"./CarBat/TripB0{j+1}.csv",sep=";",encoding='cp1252')
        lX = dfl.drop(columns=["SoC [%]", "displayed SoC [%]", "min. SoC [%]", "max. SoC [%]"])
        ly = dfl["SoC [%]"]
        trip_traces_X.append(lX)
        trip_traces_Y.append(ly)

    print(len(trip_traces_X))
    print(len(trip_traces_Y))

    X_train = df.drop(columns=["SoC [%]", "displayed SoC [%]", "min. SoC [%]", "max. SoC [%]"])
    y_train = df["SoC [%]"]

    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_regression, k=8)),
        ('regressor', DecisionTreeRegressor(random_state=69))
    ])

    pipeline.fit(X_train,y_train)

    selector = pipeline.named_steps['feature_selection']
    mask = selector.get_support()

    selected_feature_names = X_train.columns[mask]

    print("Selected features:", selected_feature_names.tolist())

    df_eval = pd.read_csv("./CarBat/TripB02.csv",sep=";", encoding='cp1252')
    X_eval = df_eval.drop(columns=["SoC [%]", "displayed SoC [%]", "min. SoC [%]", "max. SoC [%]"])
    y_eval = df_eval["SoC [%]"]

    eval_res = pipeline.predict(X_eval)
    r2 = r2_score(y_eval, eval_res)
    mse = mean_squared_error(y_eval, eval_res)
    mae = mean_absolute_error(y_eval, eval_res)

    print(f"Rsq Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    plt.figure(figsize=(12, 5))
    plt.plot(y_eval.values, label='True RUL', alpha=0.7)
    plt.plot(eval_res, label='Predicted RUL', alpha=0.7)
    plt.title("Predicted vs. True RUL")
    plt.xlabel("Sample Index")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


feature_analysis()
