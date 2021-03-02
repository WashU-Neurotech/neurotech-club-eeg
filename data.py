import numpy as np
import pandas as pd
import glob


def get_data(config):
    # Get important variables from config
    datapath = config["settings"]["data_path"]
    subjects = config["data"]["subjects"]
    electrodes = config["data"]["electrodes"]
    waves = config["data"]["waves"]

    print("Num Train:", len(glob.glob("{}Train\\*.csv".format(datapath))))
    print("Num Val:", len(glob.glob("{}Val\\*.csv".format(datapath))))
    
    # count = 0

    df_task_train = []
    df_control_train = []
    df_task_val = []
    df_control_val = []

    # training
    for subj in subjects:
        subj_files_train = glob.glob("{}Train\\{}*.csv".format(datapath, subj))
        subj_files_val = glob.glob("{}Val\\{}*.csv".format(datapath, subj))
        for f in subj_files_train:
            # count+=1
            task_type = f.split("\\")[-1].split("_")[1]
            
            print(task_type)
            temp = pd.read_csv(f, skiprows=1)

            if task_type in ["nback", "stroop", "sart", "posner"]:
                if len(df_task_train):
                    df_task_train = pd.concat([df_task_train, temp], axis=0)
                else:
                    df_task_train = temp
            elif task_type == "control":
                if len(df_control_train):
                    df_control_train = pd.concat([df_control_train, temp], axis=0)
                else:
                    df_control_train = temp
            else:
                None
                # TODO: warning

        for f in subj_files_val:
            # count+=1
            task_type = f.split("\\")[-1].split("_")[1]
            
            print(task_type)
            temp = pd.read_csv(f, skiprows=1)

            if task_type in ["nback", "stroop", "sart", "posner"]:
                if len(df_task_val):
                    df_task_val = pd.concat([df_task_val, temp], axis=0)
                else:
                    df_task_val = temp
            elif task_type == "control":
                if len(df_control_val):
                    df_control_val = pd.concat([df_control_val, temp], axis=0)
                else:
                    df_control_val = temp
            else:
                None
                # TODO: warning

    regexp = "Timestamp|"
    for electrode in electrodes:
        regexp += "EEG." + electrode + "|" # Get signal from EEG electrode (I'm assuming we will want this for every model; if we do not, add an extra hyperparameter to config)
        for wave in waves:
            regexp += "POW." + electrode + "." + wave + "|"     
    regexp = regexp[:-1] # remove the final |

    print("regex to pass to dataframe:\n\n" + regexp + "\n\n")
    
    if len(waves) > 0: # The headset picks up the waves less frequently than the raw EEG signal, drop these rows. Again, if need more control, please add hyperparam to config
        df_task_train.dropna(subset=["POW.AF3.BetaL"], inplace=True)
        df_control_train.dropna(subset=["POW.AF3.BetaL"], inplace=True)
        df_task_val.dropna(subset=["POW.AF3.BetaL"], inplace=True)
        df_control_val.dropna(subset=["POW.AF3.BetaL"], inplace=True)

    df_task_train = df_task_train.filter(regex=regexp)
    df_task_train.set_index("Timestamp", inplace=True)
    df_task_train["Label"] = 1
    print("df_task_train.head():")
    print(df_task_train.head())

    df_task_val = df_task_val.filter(regex=regexp)
    df_task_val.set_index("Timestamp", inplace=True)
    df_task_val["Label"] = 1
    print("df_task_val.head():")
    print(df_task_val.head())


    df_control_train = df_control_train.filter(regex=regexp)
    df_control_train.set_index("Timestamp", inplace=True)
    df_control_train["Label"] = 0
    print("df_control_train.head():")
    print(df_control_train.head())

    df_control_val = df_control_val.filter(regex=regexp)
    df_control_val.set_index("Timestamp", inplace=True)
    df_control_val["Label"] = 0
    print("df_control_val.head():")
    print(df_control_val.head())

    # shuffle task and no task data
    df_all_train = pd.concat([df_task_train, df_control_train], axis=0)
    df_all_train = df_all_train.sample(frac=1.0, random_state=1) # random_state sets the seed!

    df_all_val = pd.concat([df_task_val, df_control_val], axis=0)
    df_all_val = df_all_val.sample(frac=1.0, random_state=1) # random_state sets the seed!

    # normalize data
    mean = pd.concat([df_all_train.iloc[:, :-1], df_all_val.iloc[:, :-1]], axis=0).mean()
    std = pd.concat([df_all_train.iloc[:, :-1], df_all_val.iloc[:, :-1]], axis=0).std()

    df_all_train.iloc[:, :-1] = (df_all_train.iloc[:, :-1] - mean) / std
    df_all_val.iloc[:, :-1] = (df_all_val.iloc[:, :-1] - mean) / std

    print("df_all_train.head()", df_all_train.head(25))
    print("df_all_val.head()", df_all_val.head(25))

    # df_all_data = df_all.iloc[:, :-1]
    # df_all_labels = df_all.iloc[:,-1:]
    # df_all_data = (df_all_data - df_all_data.mean()) / df_all_data.std()

    # # put labels back on normalized data
    # df_all = pd.concat([df_all_data, df_all_labels], axis=1)

    # # Split into train/val sets
    # X_train = df_all.iloc[:int(0.7*len(df_all)), :-1].to_numpy()
    # Y_train = df_all.iloc[:int(0.7*len(df_all)), -1].to_numpy()

    # X_val = df_all.iloc[int(0.7*len(df_all)):, :-1].to_numpy()
    # Y_val = df_all.iloc[int(0.7*len(df_all)):, -1].to_numpy()

    X_train = df_all_train.iloc[:, :-1].to_numpy()
    Y_train = df_all_train.iloc[:, -1].to_numpy()

    X_val = df_all_val.iloc[:, :-1].to_numpy()
    Y_val = df_all_val.iloc[:, -1].to_numpy()

    print(X_train.shape)
    print(Y_train.shape)
    print(X_train)
    print(Y_train)

    print(X_val.shape)
    print(Y_val.shape)
    print(X_val)
    print(Y_val)

    return (X_train, Y_train), (X_val, Y_val)








    # for subj in subjects:
    #     subj_files = glob.glob(datapath+"{}*.csv".format(subj))
    #     for f in subj_files:
    #         count+=1
            
    #         task_type = f.split("\\")[-1].split("_")[1] # nback or control
            
    #         print(task_type)
    #         temp = pd.read_csv(f, skiprows=1)

    #         if task_type == "nback":
    #             if len(df_task):
    #                 df_task = pd.concat([df_task, temp], axis=0)
    #             else:
    #                 df_task = temp
    #         elif task_type == "control":
    #             if len(df_control):
    #                 df_control = pd.concat([df_control, temp], axis=0)
    #             else:
    #                 df_control = temp
    #         else:
    #             None
    #             # TODO: warning

    # regexp = "Timestamp|"
    # for electrode in electrodes:
    #     regexp += "EEG." + electrode + "|" # Get signal from EEG electrode (I'm assuming we will want this for every model; if we do not, add an extra hyperparameter to config)
    #     for wave in waves:
    #         regexp += "POW." + electrode + "." + wave + "|"     
    # regexp = regexp[:-1] # remove the final |

    # print("regex to pass to dataframe:\n\n" + regexp + "\n\n")
    
    # if len(waves) > 0: # The headset picks up the waves less frequently than the raw EEG signal, drop these rows. Again, if need more control, please add hyperparam to config
    #     df_task.dropna(subset=["POW.AF3.BetaL"], inplace=True)
    #     df_control.dropna(subset=["POW.AF3.BetaL"], inplace=True)

    # df_task = df_task.filter(regex=regexp)
    # df_task.set_index("Timestamp", inplace=True)
    # df_task["Label"] = 1
    # print("df_task.head():")
    # print(df_task.head())

    # df_control = df_control.filter(regex=regexp)
    # df_control.set_index("Timestamp", inplace=True)
    # df_control["Label"] = 0
    # print("df_control.head():")
    # print(df_control.head())

    # # shuffle task and no task data
    # df_all = pd.concat([df_task, df_control])
    # df_all = df_all.sample(frac=1.0, random_state=1) # random_state sets the seed!

    # # normalize data
    # df_all_data = df_all.iloc[:, :-1]
    # df_all_labels = df_all.iloc[:,-1:]
    # df_all_data = (df_all_data - df_all_data.mean()) / df_all_data.std()

    # # put labels back on normalized data
    # df_all = pd.concat([df_all_data, df_all_labels], axis=1)

    # # Split into train/val sets
    # X_train = df_all.iloc[:int(0.7*len(df_all)), :-1].to_numpy()
    # Y_train = df_all.iloc[:int(0.7*len(df_all)), -1].to_numpy()

    # X_val = df_all.iloc[int(0.7*len(df_all)):, :-1].to_numpy()
    # Y_val = df_all.iloc[int(0.7*len(df_all)):, -1].to_numpy()

    # print(X_train.shape)
    # print(Y_train.shape)

    # print(X_val.shape)
    # print(Y_val.shape)

    # return (X_train, Y_train), (X_val, Y_val)












    # # TODO: generalize and iterate over all files
    # df_task = pd.read_csv("Data/Parker/parker new nback 1_27.11.20_18.01.15.md.bp.csv", skiprows=1)
    # df_control = pd.read_csv("Data/Parker/parker control 2_25.11.20_19.19.36.md.pm.bp.csv", skiprows=1)
    # # for subj in subjects:
    # #     df_task = pd.read_csv(datapath + "/{_}")
    # df_task.dropna(subset=["POW.AF4.Alpha"], inplace=True)
    # df_control.dropna(subset=["POW.AF4.Alpha"], inplace=True)

    # # TODO: into nicer regex
    # df_task = df_task.filter(regex="Timestamp|EEG.AF|EEG.F|BetaL")
    # df_task = df_task.filter(regex="Timestamp|EEG.AF|EEG.F|POW.F|POW.AF")
    # # df_task['Timestamp'] = df_task['Timestamp']-df_task['Timestamp'].iloc[0]
    # df_task.set_index("Timestamp", inplace=True)
    # df_task["Label"] = 1

    # df_control = df_control.filter(regex="Timestamp|EEG.AF|EEG.F|BetaL")
    # df_control = df_control.filter(regex="Timestamp|EEG.AF|EEG.F|POW.F|POW.AF")
    # # df_control['Timestamp'] = df_control['Timestamp']-df_control['Timestamp'].iloc[0]
    # df_control.set_index("Timestamp", inplace=True)
    # df_control["Label"] = 0

    # # shuffle task and no task data
    # df_all = pd.concat([df_task, df_control])
    # df_all = df_all.sample(frac=1.0)

    # # normalize data
    # df_all_data = df_all.iloc[:, :-1]
    # df_all_labels = df_all.iloc[:,-1:]
    # df_all_data = (df_all_data - df_all_data.mean()) / df_all_data.std()

    # # put labels back on normalized data
    # df_all = pd.concat([df_all_data, df_all_labels], axis=1)

    # # Split into train/val sets
    # X_train = df_all.iloc[:int(0.7*len(df_all)), :-1].to_numpy()
    # Y_train = df_all.iloc[:int(0.7*len(df_all)), -1].to_numpy()

    # X_val = df_all.iloc[int(0.7*len(df_all)):, :-1].to_numpy()
    # Y_val = df_all.iloc[int(0.7*len(df_all)):, -1].to_numpy()


    # return (X_train, Y_train), (X_val, Y_val)