import numpy as np
import pandas as pd
import glob


def get_data(config):
    # Get important variables from config
    datapath = config["settings"]["data_path"]
    subjects = config["data"]["subjects"]
    electrodes = config["data"]["electrodes"]
    waves = config["data"]["waves"]

    g = Github(config["cloud"]["git_access_token"])
    repo = g.get_repo(config["cloud"]["repo"])
    contents = repo.get_contents(config["cloud"]["repo_directory"])

    controlLink = []
    taskLink = []

    for content in contents:
        if ".csv" in content.download_url:
            controlLink.append(content.download_url) if "control" in content.download_url else taskLink.append(content.download_url)
            
    df_control = pd.concat((pd.read_csv(f, skiprows = 1) for f in controlLink))
    df_task = pd.concat((pd.read_csv(f, skiprows = 1) for f in taskLink))

    regexp = "Timestamp|"
    for electrode in electrodes:
        regexp += "EEG." + electrode + "|" # Get signal from EEG electrode (I'm assuming we will want this for every model; if we do not, add an extra hyperparameter to config)
        for wave in waves:
            regexp += "POW." + electrode + "." + wave + "|"     
    regexp = regexp[:-1] # remove the final |

    print("regex to pass to dataframe:\n\n" + regexp + "\n\n")
    
    if len(waves) > 0: # The headset picks up the waves less frequently than the raw EEG signal, drop these rows. Again, if need more control, please add hyperparam to config
        df_task.dropna(subset=["POW.AF3.BetaL"], inplace=True)
        df_control.dropna(subset=["POW.AF3.BetaL"], inplace=True)

    df_task = df_task.filter(regex=regexp)
    df_task.set_index("Timestamp", inplace=True)
    df_task["Label"] = 1
    print("df_task.head():")
    print(df_task.head())

    df_control = df_control.filter(regex=regexp)
    df_control.set_index("Timestamp", inplace=True)
    df_control["Label"] = 0
    print("df_control.head():")
    print(df_control.head())

    # shuffle task and no task data
    df_all = pd.concat([df_task, df_control])
    df_all = df_all.sample(frac=1.0, random_state=1) # random_state sets the seed!

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

    print(X_train.shape)
    print(Y_train.shape)

    print(X_val.shape)
    print(Y_val.shape)

    return (X_train, Y_train), (X_val, Y_val)