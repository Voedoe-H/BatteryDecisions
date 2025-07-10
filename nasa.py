import pandas as pd
import os

data_dir = "./NASABat/cleaned_dataset/data/"

battery_data = {}

df_meta = pd.read_csv("./NASABat/cleaned_dataset/metadata.csv")

unique_types = df_meta["type"].unique()
print("Unique types:", unique_types)

#print(df_meta.head)
#print(df_meta["type"].head)
#print(df_meta["battery_id"].head)

grouped = df_meta.groupby("battery_id")

for battery, group in grouped:
    print(f"\nBattery: {battery}")
    #print(group["filename"].tolist())
    grouped_group_by_type = group.groupby("type")
    for type,type_group in grouped_group_by_type:
        print(f"{type},\n files:${type_group["filename"].to_list()}")   
        print(f"Number of {type} Files: {len(type_group["filename"].to_list())}")