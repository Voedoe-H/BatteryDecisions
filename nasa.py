import pandas as pd
import numpy as np
import os


def parse_start_time(time_str):
    try:
        time_floats = [float(i) for i in time_str.strip("[]").replace("e+03", "000").split()]
        # Convert to integers, truncate float seconds
        year, month, day, hour, minute, second = [int(x) for x in time_floats[:6]]
        return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    except:
        return pd.NaT

def generate_data():
    data_dir = "./NASABat/cleaned_dataset/data/"

    battery_data = {}
    df_meta = pd.read_csv("./NASABat/cleaned_dataset/metadata.csv")
    df_meta["parsed_time"] = df_meta["start_time"].apply(parse_start_time)

    print(df_meta["battery_id"].unique())

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

    cycles_of_battery = {}

    for battery_id, cycles in battery_cycles.items():

        battery_cycles = []
        num_cycles = len(cycles)

        for idx, (discharge_file,charge_file) in enumerate(cycles):
            try:
                discharge_path = os.path.join(data_dir,discharge_file)
                charge_path = os.path.join(data_dir,charge_file)
                
                discharge_df = pd.read_csv(discharge_path)
                charge_df = pd.read_csv(charge_path)
                
                discharge_df["Time"] = pd.to_numeric(discharge_df["Time"], errors='coerce')
                charge_df["Time"] = pd.to_numeric(charge_df["Time"], errors='coerce')

                discharge_df.dropna(subset=["Time"], inplace=True)
                charge_df.dropna(subset=["Time"], inplace=True)

                discharge_capacity = np.trapezoid(np.abs(discharge_df["Current_measured"]), discharge_df["Time"])
                charge_capacity = np.trapezoid(charge_df["Current_measured"], charge_df["Time"])
                
                discharge_duration = discharge_df["Time"].iloc[-1] - discharge_df["Time"].iloc[0]
                charge_duration = charge_df["Time"].iloc[-1] - charge_df["Time"].iloc[0]
                total_duration = discharge_duration + charge_duration

                stats = {
                    "rul": num_cycles - idx - 1,
                    "battery_id": battery_id,
                    "cycle_index": idx,
                    "discharge_capacity": discharge_capacity,
                    "charge_capacity": charge_capacity,
                    "coulombic_efficiency": abs(discharge_capacity) / abs(charge_capacity) if charge_capacity else np.nan,
                    "total_duration": total_duration,
                    "avg_voltage_discharge": discharge_df["Voltage_measured"].mean(),
                    "avg_voltage_charge": charge_df["Voltage_measured"].mean(),
                    "avg_temp_discharge": discharge_df["Temperature_measured"].mean(),
                    "avg_temp_charge": charge_df["Temperature_measured"].mean(),
                    "max_temp_discharge": discharge_df["Temperature_measured"].max(),
                    "max_temp_charge": charge_df["Temperature_measured"].max(),
                }
                battery_cycles.append(stats)
            except Exception as e:
                print(f"Error for {idx} with exception: {e} ")
            
            cycles_of_battery[battery_id] = pd.DataFrame(battery_cycles)

    rolling_features = [
            "discharge_capacity", "charge_capacity", "coulombic_efficiency",
            "avg_voltage_discharge", "avg_voltage_charge",
            "avg_temp_discharge", "avg_temp_charge"
    ]

    for id,cycles in cycles_of_battery.items():
        #print(cycles.head)
        for feature in rolling_features:
            cycles[f"{feature}_rolling_mean"] = cycles[feature].rolling(window = 5,min_periods=1).mean()
    
    
    #print(cycles_of_battery["B0045"])
    
    for id,cycles in cycles_of_battery.items():
        cycles.to_csv(f"./mydat/{id}")
    
def created_data_set_analysis():
    data_set_dir = "./mydat"
    data_frames = []
    for x in os.listdir(data_set_dir):
        df = pd.read_csv(f"{data_set_dir}/{x}")
        print(df.head)

if __name__ == "__main__":
    generate_data()