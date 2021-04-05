# Neurotech Club GitHub

This repo contains WashU Neurotech Club's code for our EEG BCI.

## Installation

Create a conda environment

TODO: create a neurotech_club_env.yml with necessary packages and add to this GitHub!! THE FOLLOWING WILL NOT DO ANYTHING YET! 
```bash
conda env create -f neurotech_club_env.yml
```

## Usage
Please set all hyperparameters for the model in config.json. When creating new model types and modifying old ones,
please ensure that all hyperparameters of interest can be adjusted in config.json so that there is only ONE place 
to do so. This may be a bit annoying, but the config.json files are saved in results/saved_models/configs so that we
know exactly how we built a specific model.

Possible values for EEG data (please write EXACTLY as below: there are currently no checks in code for this, so a typo may break it!!):
* electrodes : AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
* waves (can be an empty array if you only want raw signal): Theta, Alpha, BetaL, BetaH, Gamma

Cloud Stuff:
Put the repo name in the repo thing. 
If csv's are stores in root repo directory, then leave repo_repository blank. 
Remember to get your own git access token from Settings => Developer Settings, then paste it in the space given

```json

{
    "settings": {
        "root_path": "Directory in which main.py is stored",
        "data_path": "Directory in which data is stored (file names must follow Neurotech Club's conventions)"
    },

    "data": {
        "subjects": ["array", "of", "subjects"],
        "electrodes": ["array", "of", "electrodes"],
        "waves": ["array", "of", "waves"]
    },

    "model": {
        "model_type": "name_of_model_to_use_see_below",
        "baby_binary": {
            "num_hidden_layer_nodes": <int_num_nodes>
        },
        "dnn_binary": {
            "hidden_layers": [<int_num_nodes_layer_1>, <int_num_nodes_layer_2>, ...]
        }
    },

    "train": {
        "epochs": <int_num_epochs>,
        "learning_rate": <int_learning_rate>
    },

    "cloud":{
        "repo": "v-puppala/Neurotech-Data",
        "repo_directory": "OneDrive-2021-02-13",
        "git_access_token": "Get Your Own From Github Settings => Developer Settings, then paste in this space"
    }
}
```
Once the hyperparameters are set to your desire, run the following.
```bash
python main.py
```