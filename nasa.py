import pandas as pd
import os


def parse_start_time(time_str):
    try:
        time_floats = [float(i) for i in time_str.strip("[]").replace("e+03", "000").split()]
        return pd.to_datetime(time_floats[:6], errors='coerce')  # year, month, day, hour, minute, second
    except:
        return pd.NaT


data_dir = "./NASABat/cleaned_dataset/data/"

battery_data = {}

df_meta = pd.read_csv("./NASABat/cleaned_dataset/metadata.csv")
df_meta["parsed_time"] = df_meta["start_time"].apply(parse_start_time)
unique_types = df_meta["type"].unique()
print("Unique types:", unique_types)

#print(df_meta.head)
#print(df_meta["type"].head)
#print(df_meta["battery_id"].head)
# Impedance File Structure: Sense_current,Battery_current,Current_ratio,Battery_impedance,Rectified_Impedance
# Discharge File Structure: Voltage_measured,Current_measured,Temperature_measured,Current_load,Voltage_load,Time
# Charge File Structure: Voltage_measured,Current_measured,Temperature_measured,Current_charge,Voltage_charge,Time

grouped = df_meta.groupby("battery_id")
unique_ids = df_meta["battery_id"].unique()
print(f"Number of batteries:{len(grouped)}")

for battery, group in grouped:
    print(f"\nBattery: {battery}")
    #print(group["filename"].tolist())
    grouped_group_by_type = group.groupby("type")
    for type,type_group in grouped_group_by_type:
        print(f"{type},\n files:${type_group["filename"].to_list()}")   
        print(f"Number of {type} Files: {len(type_group["filename"].to_list())}")