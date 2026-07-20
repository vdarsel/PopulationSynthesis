import pandas as pd
from model_launcher import Model_for_Population_Synthesis

from utils.utils_dir import get_folder_evaluation

def scores_comparison(config, models:list[Model_for_Population_Synthesis]):
    metrics = ["\overline{SRMSE}_3", "SSCIOD", "Wasserstein-DCR"]
    scores = pd.DataFrame(columns=metrics)
    for model in models:
        name = model.termination_saving()
        dir_path_evaluation_generated_data = get_folder_evaluation(config, name)
        scores_model = pd.read_csv(os.path.join(dir_path_evaluation_generated_data, "overview_score.csv"), sep=";", index_col="Metric")
        scores.loc[name[1:].replace("_"," "), metrics] = scores_model.loc[metrics, f"{config.attributes_setname}_{config.size_data_str}{name}"]
    print(scores)
    scores.to_csv(os.path.join(config.sample_folder, config.variable, config.size_data_str, "quick_comparison.csv"), sep=";")