import numpy as np
import pandas as pd
import os

from evaluation.proportion_sampling import generate_proportion_from_dataset
from evaluation.metrics_proportion import get_df_scores_by_cat, get_scores_agg
from evaluation.heatmap import generate_color_map_save, generate_color_map_filter_save
from evaluation.metrics_privacy import Distance_to_Closest_Records,generate_histogram_DCR
from evaluation.metrics_originality import get_proportion_from_original_data_df, get_proportion_from_original_data_df_not_in_other_df, get_rate_of_impossible_combinations

import matplotlib.pyplot as plt
import plotly.graph_objects as go

CAT_RARE_VALUE = '__rare__'

def preprocessing_cat_data_dataframe_sampling(dataset_ref: pd.DataFrame, min_count, cols_cat, other_dataset : list[pd.DataFrame] = []):
    if(len(cols_cat)>0):
        X_new = dataset_ref.copy()
        X_new_list = [df.copy() for df in other_dataset]
        print("Initial categories (training):", dataset_ref[cols_cat].nunique().to_list())
        for column_idx in (cols_cat):
            value_counts = dataset_ref[column_idx].value_counts()
            popular_categories = value_counts[value_counts>=min_count].index
            X_new.loc[~X_new[column_idx].isin(popular_categories),column_idx] = CAT_RARE_VALUE
            for df in X_new_list:
                df.loc[~df[column_idx].isin(popular_categories),column_idx] = CAT_RARE_VALUE
        print("Final categories (training):", X_new[cols_cat].nunique().to_list())
    return X_new, X_new_list


def generate_histogram(df, path, i):
    metrics = ["SRMSE","Hellinger","Pearson","R2"]
    for metric in metrics:
        plt.figure(figsize=[7,7])
        plt.hist(df[metric])
        plt.title(f'{metric} {i}')
        plt.savefig(f"{path}/hist_{metric}_{i}.png")
        plt.close()
    
def generate_plot(proportion_sample, proportion_ref, path, i):
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    plt.figure(figsize=[7,7])
    plt.plot([-1,1], [-1,1], color='r', alpha=0.5)
    plt.scatter(proportion_ref,proportion_sample)
    plt.xlim(-0.02,np.max([proportion_ref,proportion_sample])+0.02)
    plt.ylim(-0.02,np.max([proportion_ref,proportion_sample])+0.02)
    plt.xlabel("Original data proportion")
    plt.ylabel("Sampled data proportion")
    plt.title(dict_title[i])
    plt.savefig(f"{path}/comparison_{i}.png")
    plt.close()
    generate_color_map_save(proportion_ref,proportion_sample,path,f"Full_Heatmap_{i}",200)
    generate_color_map_filter_save(proportion_ref,proportion_sample,path,f"Full_Heatmap_filter_{i}",2000, min_freq=1)

def generate_plot_plotly(proportion_sample, proportion_ref, combi, values, columns_name, path, i, target=100000):
    coef = max(len(proportion_sample)//target,1)
    keep_idx = np.array([i%coef==0 for i in range(len(proportion_ref))])
    proportion_ref = proportion_ref[keep_idx]
    proportion_sample = proportion_sample[keep_idx]
    combi = combi[keep_idx]
    values = values[keep_idx]
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    fig = go.Figure(data=[go.Scatter(x=[-1,1], y=[-1,1],mode="lines",line=dict(color='rgba(255, 17, 0, 0.5)'))],
                    layout=go.Layout(title=dict_title[i], showlegend=False,
                                     xaxis_range=[-0.02,np.max([proportion_ref,proportion_sample])+0.02],
                                     yaxis_range=[-0.02,np.max([proportion_ref,proportion_sample])+0.02],
                                     xaxis_title="Original data proportion",
                                     yaxis_title="Sampled data proportion"))
    fig.add_trace(go.Scatter(x=proportion_ref,y=proportion_sample,mode="markers", customdata=np.stack([columns_name[combi.astype(int)],values], axis=1),
                             hovertemplate="<br>".join([
                                "ColX: %{x}",
                                "ColY: %{y}",
                                "Col1: %{customdata[0]}",
                                "Col2: %{customdata[1]}",
                            ])))
    fig.write_html(f"{path}/comparison_{i}.html")
    generate_color_map_save(proportion_ref,proportion_sample,path,f"Full_Heatmap_{i}",2000)
    generate_color_map_filter_save(proportion_ref,proportion_sample,path,f"Full_Heatmap_filter_{i}",2000, min_freq=10)



def full_evaluation(args, term_evaluation=""):
    
    ##################
    ### Parameters ###
    ##################
        
    if not ("special_model" in args.__dict__.keys()):
        args.special_model= ""
    
    dir_path_generated_data = f"{args.sample_folder}/{args.folder_save+term_evaluation}"
    
    dir_path_evaluation_generated_data = f"{dir_path_generated_data}/{args.dataset_evaluation}"
    
    datapath = "Data"
    
    filename_training = args.filename_training
    
    if not os.path.isdir(dir_path_evaluation_generated_data):
        os.makedirs(dir_path_evaluation_generated_data)
    n_samples = args.n_generation
    
    filename_sampling = filename_training.split(".")[0]+(args.sampling_terminaison+"_"+str(n_samples)+term_evaluation)
    basename = f"{dir_path_generated_data}/{filename_sampling}"
    folder_proportion_file_generated_data = f"{dir_path_evaluation_generated_data}/Proportion_save/"
    if not os.path.isdir(folder_proportion_file_generated_data):
        os.makedirs(folder_proportion_file_generated_data)
    basename_save = f"{folder_proportion_file_generated_data}/{filename_training.split(".csv")[0]+args.sampling_terminaison}_{str(args.n_generation)+term_evaluation}"
    file_test = f"{datapath}/{args.dataset_evaluation}/{args.filename_test.split(".csv")[0]+args.special_model+".csv"}"
    file_test_WDCR = f"{datapath}/{args.dataset_evaluation}/{args.filename_test_WDCR.split(".csv")[0]+args.special_model+".csv"}"
    file_train = f"{datapath}/{args.dataname}/{filename_training}"
        
    ########################
    ### Initiate results ###
    ########################
    
    Ser_review = pd.Series(index=["\overline{SRMSE}_1","\overline{SRMSE}_2","\overline{SRMSE}_3",
                     "\overline{Hellinger}_1","\overline{Hellinger}_2","\overline{Hellinger}_3",
                     "SRMSE_1","SRMSE_2","SRMSE_3",
                     "Hellinger_1","Hellinger_2","Hellinger_3",
                     "Pearson_1","Pearson_2","Pearson_3",
                     "R2_1","R2_2","R2_3",
                     "Rate of copies (training)", "Rate of copies (training, without geo)",
                     "Rate of copies (in testing not in training)", "Rate of copies (in testing not in training, without geo)",
                     "SSCIOD", "SSCIOD (without geo)"
                     ], dtype=str)

    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    df_scores_median = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    df_scores_mean = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    df_scores_agg = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    
    #################
    ### Load data ###
    #################

    df_info = pd.read_csv(f"{datapath}/{args.dataname}/{args.infoname}",sep=";")
    df_info = df_info[df_info[args.attributes_setname]]

    columns = df_info["Variable_name"]
    columns_without_geo = df_info[(~df_info["Geographical_attribute"])]["Variable_name"]

    idx_num = np.arange(len(df_info))[(df_info["Type"].isin(["int","float"]))]
    name_cat = df_info["Variable_name"][(df_info["Type"].isin(["binary","boolean","category"]))].to_list()

    dataset_train = pd.read_csv(file_train, sep=";", low_memory=False)[df_info["Variable_name"]]

    for idx in df_info[df_info["Type"].isin(["category"])]["Variable_name"]:
        dataset_train[idx] = dataset_train[idx].astype(str)
    dataset_train = dataset_train[columns]

    dataset_test = pd.read_csv(file_test, sep=";", low_memory=False)
    dataset_test_WDCR = pd.read_csv(file_test_WDCR, sep=";", low_memory=False)

    for idx in df_info[(df_info["Type"].isin(["category"]))]["Variable_name"]:
        dataset_test[idx] = dataset_test[idx].astype(str)
    dataset_test = dataset_test[columns]

    min_size_category = args.transform.cat_min_count
    dataset_train,list_df  = preprocessing_cat_data_dataframe_sampling(dataset_train, min_size_category, name_cat, [dataset_test])
    dataset_test = list_df[0][columns]
    dataset_test_numpy = dataset_test.to_numpy()
    
    dataset_train_numpy = dataset_train.to_numpy()
    

    
    df_sample = pd.read_csv(f'{basename}.csv', sep=";")

    for col in df_info[(df_info["Type"].isin(["category"]))]["Variable_name"]:
        df_sample[col] = df_sample[col].astype(str)
    df_sample = df_sample[columns]
    
    if (args.dataset_evaluation!=args.dataname): 
        # For future functionalities
        folder_test_file = f"results/Proportion_save/{args.attributes_setname}/{args.dataset_evaluation}/{args.dataname}"
        name_test_file_distribution = f"Proportion_test_data_distribution_{args.dataset_evaluation}_reference_data_{args.dataname}_{(filename_training).split(".")[0]}_{args.attributes_setname}_{args.filename_test.split(".csv")[0]+args.special_model}"
        name_test_file_realism = f"Proportion_test_data_realism_{args.dataset_evaluation}_reference_data_{args.dataname}_{(filename_training).split(".")[0]}_{args.attributes_setname}_{args.filename_test.split(".csv")[0]+args.special_model}"
    else:
        folder_test_file = f"results/Proportion_save/{args.attributes_setname}/{args.dataname}/{args.dataname}"
        name_test_file_distribution = f"Proportion_reference_data_distribution_{args.dataname}_{(filename_training).split(".")[0]}_{args.attributes_setname}_{args.filename_test.split(".csv")[0]+args.special_model}"
        name_test_file_realism = f"Proportion_reference_data_realism_{args.dataname}_{(filename_training).split(".")[0]}_{args.attributes_setname}_{args.filename_test.split(".csv")[0]+args.special_model}"

    ######################################
    ### Evaluation of the distribution ###
    ######################################

    for i in range(1,4):    
        basename_save_distribution = f"{basename_save}_distribution"
        proportion_file = f"{basename_save_distribution}_{i}.npy"
        dataset_test_numpy_distribution = np.copy(dataset_test_numpy)
        dataset_generated_distribution = df_sample.to_numpy()
        # bins_distribution = df_info["Bin_distribution"].to_list()
        # for j in idx_num:
        #     dataset_test_numpy_distribution[:,j] = (dataset_test_numpy_distribution[:,j]//bins_distribution[j])*bins_distribution[j]
        #     dataset_generated_distribution[:,j] = (dataset_generated_distribution[:,j]//bins_distribution[j])*bins_distribution[j]
        if (not os.path.isfile(proportion_file)):
            print("\nGenerate Proportion file for sample (distribution)")
            generate_proportion_from_dataset(dataset_test_numpy_distribution,dataset_generated_distribution,i, '.', basename_save_distribution,False)
        if (not os.path.isfile(f"{folder_test_file}/{name_test_file_distribution}_{i}.npy")):
            if(not os.path.isdir(folder_test_file)):
                os.makedirs(folder_test_file)
            print("\nGenerate Proportion file for test data (distribution)")
            generate_proportion_from_dataset(dataset_test_numpy_distribution,dataset_test_numpy_distribution,i, folder_test_file, name_test_file_distribution,True)
        proportion = np.load(proportion_file)
        combi_test_file = f"{folder_test_file}/{name_test_file_distribution}_{i}_comb.npy"
        value_test_file = f"{folder_test_file}/{name_test_file_distribution}_{i}_values.npy"
        proportion_test = np.load(f"{folder_test_file}/{name_test_file_distribution}_{i}.npy")
        combi_test = np.load(combi_test_file)
        values_test = np.load(value_test_file, allow_pickle=True)
        df_scores_by_cat = get_df_scores_by_cat(proportion,proportion_test,combi_test,i)
        df_scores_by_cat.to_csv(f"{dir_path_evaluation_generated_data}/scores_by_cat_{i}.csv", sep=";")
        generate_histogram(df_scores_by_cat, dir_path_evaluation_generated_data, i)
        generate_plot(proportion, proportion_test, dir_path_evaluation_generated_data, i)
        generate_plot_plotly(proportion, proportion_test, combi_test, values_test, columns.to_numpy(),dir_path_evaluation_generated_data, i)
        df_scores_median.loc[dict_title[i]] =  df_scores_by_cat[["SRMSE","Hellinger","Pearson","R2"]].median()
        df_scores_mean.loc[dict_title[i]] = df_scores_by_cat[["SRMSE","Hellinger","Pearson","R2"]].mean()
        df_scores_agg.loc[dict_title[i]] = pd.Series(get_scores_agg(proportion_test,proportion),index=["SRMSE","Hellinger","Pearson","R2"])

    #################################
    ### Evaluation of the realism ###
    #################################

    rate_non_geo_list, rate_geo_list = [],[]
    for i in range(1,4):
        basename_save_realism = f"{basename_save}_realism"
        proportion_file = f"{basename_save_realism}_{i}.npy"
        dataset_generated_realism = df_sample.to_numpy()
        if (not os.path.isfile(proportion_file)):
            print("\nGenerate Proportion file for sample (realism)")
            generate_proportion_from_dataset(dataset_test_numpy_distribution,dataset_generated_realism,i, '.', basename_save_realism,False)
        if (not os.path.isfile(f"{folder_test_file}/{name_test_file_realism}_{i}.npy")):
            if(not os.path.isdir(folder_test_file)):
                os.makedirs(folder_test_file)
            print("\nGenerate Proportion file for test data (realism)")
            generate_proportion_from_dataset(dataset_test_numpy_distribution,dataset_test_numpy_distribution,i, folder_test_file, name_test_file_realism,True)

        proportion = np.load(proportion_file)
        proportion_test = np.load(f"{folder_test_file}/{name_test_file_realism}_{i}.npy")
        values = np.load(f"{folder_test_file}/{name_test_file_realism}_{i}_values.npy", allow_pickle=True)
        combs = np.load(f"{folder_test_file}/{name_test_file_realism}_{i}_comb.npy")
        rate_non_geo, rate_geo = get_rate_of_impossible_combinations(df_sample,df_info, proportion_test, proportion, columns, combs, values)
        rate_non_geo_list.append(rate_non_geo)
        rate_geo_list.append(rate_geo)
        
    #################################
    ### Evaluation of the privacy ###
    #################################

    if (not os.path.isfile(f"{dir_path_evaluation_generated_data}/dcr.csv")):
        df_DCR = Distance_to_Closest_Records(df_sample,dataset_train,dataset_test_WDCR, df_info.reset_index(), f"{folder_test_file}/{name_test_file_distribution}")
        df_DCR.to_csv(f"{dir_path_evaluation_generated_data}/dcr.csv", sep=";",index=False)
    df_DCR = pd.read_csv(f"{dir_path_evaluation_generated_data}/dcr.csv", sep=";")
    generate_histogram_DCR(df_DCR, dir_path_evaluation_generated_data)
    Wasserstein_distance_DCR = np.sqrt(1/(len(df_DCR)**2)*np.sum(np.power(df_DCR["DCR train"].sort_values().values - df_DCR["DCR test"].sort_values().values,2)))
        
    ############################
    ### Creation of csv file ###
    ############################
        
    Ser_review["\overline{SRMSE}_1"] = f"{df_scores_mean["SRMSE"]["Marginal"] :.3g}"
    Ser_review["\overline{SRMSE}_2"] = f"{df_scores_mean["SRMSE"]["Bivariate"] :.3g}"
    Ser_review["\overline{SRMSE}_3"] = f"{df_scores_mean["SRMSE"]["Trivariate"] :.3g}"
    Ser_review["\overline{Hellinger}_1"] = f"{df_scores_mean["Hellinger"]["Marginal"] :.3g}"
    Ser_review["\overline{Hellinger}_2"] = f"{df_scores_mean["Hellinger"]["Bivariate"] :.3g}"
    Ser_review["\overline{Hellinger}_3"] = f"{df_scores_mean["Hellinger"]["Trivariate"] :.3g}"
    Ser_review["SRMSE_1"] = f"{df_scores_agg["SRMSE"]["Marginal"] :.3g}"
    Ser_review["SRMSE_2"] = f"{df_scores_agg["SRMSE"]["Bivariate"] :.3g}"
    Ser_review["SRMSE_3"] = f"{df_scores_agg["SRMSE"]["Trivariate"] :.3g}"
    Ser_review["Hellinger_1"] = f"{df_scores_agg["Hellinger"]["Marginal"] :.3g}"
    Ser_review["Hellinger_2"] = f"{df_scores_agg["Hellinger"]["Bivariate"] :.3g}"
    Ser_review["Hellinger_3"] = f"{df_scores_agg["Hellinger"]["Trivariate"] :.3g}"
    Ser_review["Pearson_1"] = f"{df_scores_agg["Pearson"]["Marginal"] :.3g}"
    Ser_review["Pearson_2"] = f"{df_scores_agg["Pearson"]["Bivariate"] :.3g}"
    Ser_review["Pearson_3"] = f"{df_scores_agg["Pearson"]["Trivariate"] :.3g}"
    Ser_review["R2_1"] = f"{df_scores_agg["R2"]["Marginal"] :.3g}"
    Ser_review["R2_2"] = f"{df_scores_agg["R2"]["Bivariate"] :.3g}"
    Ser_review["R2_3"] = f"{df_scores_agg["R2"]["Trivariate"] :.3g}"
    Ser_review["SSCIOD"] = f'{rate_geo_list[1]:.2%}'
    Ser_review["SSCIOD (without geo)"] = f'{rate_non_geo_list[1]:.2%}'
    Ser_review["Rate of copies (training)"] = f'{np.abs(get_proportion_from_original_data_df(dataset_train,dataset_test,columns)-get_proportion_from_original_data_df(dataset_train,df_sample,columns)):.2%}'
    Ser_review["Rate of copies (training, without geo)"] = f'{np.abs(get_proportion_from_original_data_df(dataset_train,dataset_test,columns_without_geo)-get_proportion_from_original_data_df(dataset_train,df_sample,columns_without_geo)):.2%}'
    val = get_proportion_from_original_data_df_not_in_other_df(dataset_test,dataset_train,df_sample,columns)
    if (val=="NA"):
        Ser_review["Rate of copies (in testing not in training)"] = 'NA'
    else:
        Ser_review["Rate of copies (in testing not in training)"] = f'{val:.2%}'
    val = get_proportion_from_original_data_df_not_in_other_df(dataset_test,dataset_train,df_sample,columns_without_geo)
    if (val=="NA"):
        Ser_review["Rate of copies (in testing not in training, without geo)"] = 'NA'
    else:
        Ser_review["Rate of copies (in testing not in training, without geo)"] = f'{val:.2%}'
    Ser_review["Wasserstein-DCR"] = Wasserstein_distance_DCR

    Ser_review.rename(f"{args.attributes_setname}_{args.folder_save_end}{term_evaluation}").to_csv(f"{dir_path_evaluation_generated_data}/overview_score.csv", sep=";", index_label="Metric")
