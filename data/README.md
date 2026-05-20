# French Census data and the dataset's name.
To run the experiments with the French Census data, please refer to the detailed dataset documentation in [this folder](https://github.com/vdarsel/PopulationSynthesis/tree/main/data/French_Census_data_2021). Instructions on how to download the data can be found at the [the root of this project](https://github.com/vdarsel/PopulationSynthesis/tree/main/).


# Other dataset

This code can be adapted to any dataset by adding the new data to this folder in your local version.
There are only two elements you need to add or modify:
- Create an ````info.csv```` file tailored to your dataset of interest. This file lists the variables, their types, their corresponding scenarios, etc.
- Modify the configuration files in [conf](https://github.com/vdarsel/PopulationSynthesis/tree/main/conf). ```conf_size``` handles the name of the files (which may not require any change), ```conf_variable``` holds the folders' names and the dataset's name.