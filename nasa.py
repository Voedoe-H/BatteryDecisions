import pandas as pd
import os


def parse_start_time(time_str):
    try:
        time_floats = [float(i) for i in time_str.strip("[]").replace("e+03", "000").split()]
        # Convert to integers, truncate float seconds
        year, month, day, hour, minute, second = [int(x) for x in time_floats[:6]]
        return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    except:
        return pd.NaT

data_dir = "./NASABat/cleaned_dataset/data/"

battery_data = {}

df_meta = pd.read_csv("./NASABat/cleaned_dataset/metadata.csv")
df_meta["parsed_time"] = df_meta["start_time"].apply(parse_start_time)

# Impedance File Structure: Sense_current,Battery_current,Current_ratio,Battery_impedance,Rectified_Impedance
# Discharge File Structure: Voltage_measured,Current_measured,Temperature_measured,Current_load,Voltage_load,Time
# Charge File Structure: Voltage_measured,Current_measured,Temperature_measured,Current_charge,Voltage_charge,Time

grouped = df_meta.groupby("battery_id")

for battery, group in grouped:
    #print(f"\nBattery: {battery}")
    ordered_files = group.sort_values(by=["parsed_time"])
    grouped_group_by_type = ordered_files.groupby("type")

    for type,type_group in grouped_group_by_type:
        pass
        #print(f"{type},\n files:${type_group["filename"].to_list()}")   
        #print(f"Number of {type} Files: {len(type_group["filename"].to_list())}")


battery_cycles = {}

for battery_id, group in df_meta.groupby("battery_id"):

    ordered = group.sort_values(by="parsed_time").reset_index(drop=True)
    
    cycles = []
    i = 0
    while i < len(ordered) - 1:
        current_row = ordered.iloc[i]
        next_row = ordered.iloc[i + 1]
        
        if current_row["type"] == "discharge" and next_row["type"] == "charge":
            cycles.append((current_row["filename"], next_row["filename"]))
            i += 2
        else:
            i += 1

    battery_cycles[battery_id] = cycles
    #print(f"Battery {battery_id} — Found {len(cycles)} cycles")

for battery_id, cycles in battery_cycles.items():
    for idx, (discharge_file,charge_file) in enumerate(cycles):
        try:
            discharge_path = os.path.join(data_dir,discharge_file)
            charge_path = os.path.join(data_dir,charge_file)
            
            discharge_df = pd.read_csv(discharge_path)
            charge_df = pd.read_csv(charge_path)
            print(discharge_df.head)
            print(charge_df.head)

        except Exception as e:
            print(f"Error for {idx} with exception: {e} ")
        
    break