import numpy as np
import pandas as pd


def get_data(config):
    subjects = config["data"]["subjects"]
    datapath = config["settings"]["datapath"]
    # TODO: generalize and iterate over all files
    df_task = pd.read_csv("Data/Parker/parker new nback 1_27.11.20_18.01.15.md.bp.csv", skiprows=1)
    df_control = pd.read_csv("Data/Parker/parker control 2_25.11.20_19.19.36.md.pm.bp.csv", skiprows=1)
    # for subj in subjects:
    #     df_task = pd.read_csv(datapath + "/{_}")
    df_task.dropna(subset=["POW.AF4.Alpha"], inplace=True)
    df_control.dropna(subset=["POW.AF4.Alpha"], inplace=True)

    # TODO: into nicer regex
    df_task = df_task.filter(regex="Timestamp|EEG.AF|EEG.F|BetaL")
    df_task = df_task.filter(regex="Timestamp|EEG.AF|EEG.F|POW.F|POW.AF")
    # df_task['Timestamp'] = df_task['Timestamp']-df_task['Timestamp'].iloc[0]
    df_task.set_index("Timestamp", inplace=True)
    df_task["Label"] = 1

    df_control = df_control.filter(regex="Timestamp|EEG.AF|EEG.F|BetaL")
    df_control = df_control.filter(regex="Timestamp|EEG.AF|EEG.F|POW.F|POW.AF")
    # df_control['Timestamp'] = df_control['Timestamp']-df_control['Timestamp'].iloc[0]
    df_control.set_index("Timestamp", inplace=True)
    df_control["Label"] = 0

    # shuffle task and no task data
    df_all = pd.concat([df_task, df_control])
    df_all = df_all.sample(frac=1.0)

    # normalize data
    df_all_data = df_all.iloc[:, :-1]
    df_all_labels = df_all.iloc[:,-1:]
    df_all_data = (df_all_data - df_all_data.mean()) / df_all_data.std()

    # put labels back on normalized data
    df_all = pd.concat([df_all_data, df_all_labels], axis=1)

    # Split into train/val sets
    X_train = df_all.iloc[:int(0.7*len(df_all)), :-1].to_numpy()
    Y_train = df_all.iloc[:int(0.7*len(df_all)), -1].to_numpy()

    X_val = df_all.iloc[int(0.7*len(df_all)):, :-1].to_numpy()
    Y_val = df_all.iloc[int(0.7*len(df_all)):, -1].to_numpy()


    return (X_train, Y_train), (X_val, Y_val)