import numpy as np
import pandas as pd
from github import Github


def get_data(config):

    g = Github(config["settings"]["git_access_token"])
    repo = g.get_repo(config["settings"]["repo"])
    contents = repo.get_contents(config["settings"]["repo_directory"])

    controlLink = []
    taskLink = []

    for content in contents:
        if ".csv" in content.download_url:
            controlLink.append(content.download_url) if "control" in content.download_url else taskLink.append(content.download_url)
            
    df_control = pd.concat((pd.read_csv(f, skiprows = 1) for f in controlLink))
    df_task = pd.concat((pd.read_csv(f, skiprows = 1) for f in taskLink))

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