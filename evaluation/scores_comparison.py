import pandas as pd
from model_launcher import Model_for_Population_Synthesis

def scores_comparison(args, models:list[Model_for_Population_Synthesis]):
    metrics = ["\overline{SRMSE}_3", "SSCIOD", "Wasserstein-DCR"]
    scores = pd.DataFrame(columns=metrics)
    for model in models:
        name = model.terminaison_saving()
        dir_path_generated_data = f"{args.sample_folder}/{args.folder_save+name}"
        dir_path_evaluation_generated_data = f"{dir_path_generated_data}/Proportion_save/{args.dataset_evaluation}"
        scores_model = pd.read_csv(f"{dir_path_evaluation_generated_data}/overview_score.csv", sep=";", index_col="Metric")
        scores.loc[name[1:].replace("_"," "), metrics] = scores_model.loc[metrics, f"{args.attributes_setname}_{args.folder_save_end}{name}"]
    print(scores)
    scores.to_csv(f"{args.sample_folder}/quick_comparison.csv", sep=";")
        
        