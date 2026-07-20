import numpy as np
import pandas as pd
import os

from evaluation.proportion_sampling import compute_proportion_file_from_unique_array_and_df, recover_lists_from_dictionnary, adapt_numerical_values
from evaluation.metrics_proportion import get_df_scores_by_cat, get_scores_agg
from evaluation.heatmap import generate_color_map_save, generate_color_map_filter_save
from evaluation.metrics_privacy import Distance_to_Closest_Records,generate_histogram_DCR
from evaluation.metrics_originality import get_proportion_from_original_data_df, get_proportion_from_original_data_df_not_in_other_df, get_rate_of_impossible_combinations

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils.utils_import import get_info_file
from utils.utils_dir import get_file_path_sampling, get_data_dir, get_testing_data_dir, get_folder_evaluation, get_filename_sampling

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
        plt.savefig(os.path.join(path, "hist_" + metric + "_" + i + ".png"))
        plt.close()
    
def generate_plot(proportion_sample, proportion_ref, path, i):
    proportion_ref_concat = np.concat(proportion_ref)
    proportion_sample_concat = np.concat(proportion_sample)
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    plt.figure(figsize=[7,7])
    plt.plot([-1,1], [-1,1], color='r', alpha=0.5)
    for p,q in zip(proportion_ref, proportion_sample):
        plt.scatter(p,q)
    plt.xlim(-0.02,np.max([proportion_ref_concat,proportion_sample_concat])+0.02)
    plt.ylim(-0.02,np.max([proportion_ref_concat,proportion_sample_concat])+0.02)
    plt.xlabel("Original data proportion")
    plt.ylabel("Sampled data proportion")
    plt.title(dict_title[i])
    plt.savefig(os.path.join(path,"comparison_"+i+".png"))
    plt.close()
    generate_color_map_save(proportion_ref_concat,proportion_sample_concat,path,f"Full_Heatmap_{i}",200)
    generate_color_map_filter_save(proportion_ref_concat,proportion_sample_concat,path,f"Full_Heatmap_filter_{i}",2000, min_freq=1)

def generate_plot_plotly(proportion_sample, proportion_ref, combi_names, values, path, i, target=100000):
    coef = max(len(proportion_sample)//target,1)
    keep_idx = np.array([i%coef==0 for i in range(len(proportion_ref))])
    proportion_ref = proportion_ref[keep_idx]
    proportion_sample = proportion_sample[keep_idx]
    combi_names = combi_names[keep_idx]
    values = values[keep_idx]
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    fig = go.Figure(data=[go.Scatter(x=[-1,1], y=[-1,1],mode="lines",line=dict(color='rgba(255, 17, 0, 0.5)'))],
                    layout=go.Layout(title=dict_title[i], showlegend=False,
                                     xaxis_range=[-0.02,np.max([proportion_ref,proportion_sample])+0.02],
                                     yaxis_range=[-0.02,np.max([proportion_ref,proportion_sample])+0.02],
                                     xaxis_title="Original data proportion",
                                     yaxis_title="Sampled data proportion"))
    fig.add_trace(go.Scatter(x=proportion_ref,y=proportion_sample,mode="markers", customdata=np.stack([combi_names,values], axis=1),
                             hovertemplate="<br>".join([
                                "ColX: %{x}",
                                "ColY: %{y}",
                                "Col1: %{customdata[0]}",
                                "Col2: %{customdata[1]}",
                            ])))
    fig.write_html(os.path.join(path, "comparison_"+i+".html"))
    generate_color_map_save(proportion_ref,proportion_sample,path,f"Full_Heatmap_{i}",2000)
    generate_color_map_filter_save(proportion_ref,proportion_sample,path,f"Full_Heatmap_filter_{i}",2000, min_freq=10)


def output_serie_eval(dict_unique_values, basename_file_save, reference_DCR_file, proportion_test_file, df_for_evaluation, dataset_test_df, dataset_test_df_dcr, dataset_reference_training, df_info, dir_path_save_results=None, save=False):
    
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

    columns = df_info["Variable_name"].to_numpy()
    columns_without_geo = df_info[(~df_info["Geographical_attribute"])]["Variable_name"]

    
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    df_scores_median = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    df_scores_mean = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    df_scores_agg = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])

    for i in range(1,4):    
        proportion_file = f"{basename_file_save}_{i}.npy"
        if (not os.path.isfile(proportion_file)):
            print("\nGenerate Proportion file for sample")
            
            df_for_evaluation_temp = adapt_numerical_values(df_for_evaluation,dict_unique_values, df_info)
            compute_proportion_file_from_unique_array_and_df(dict_unique_values,
                    df_for_evaluation_temp.astype(str),
                    columns,
                    basename_file_save,
                    i,
                    ".")
        combi_test_file = f"{proportion_test_file}_{i}_comb.npy"
        value_test_file = f"{proportion_test_file}_{i}_values.npy"

        proportion_concat = np.load(proportion_file)
        proportion_test_concat = np.load(f"{proportion_test_file}_{i}.npy")
        combi_concat = np.load(combi_test_file).astype(int)
        combi_names_concat = columns[combi_concat]
        values_test_concat = np.load(value_test_file, allow_pickle=True)
        
        proportion = recover_lists_from_dictionnary(columns, dict_unique_values, proportion_concat, i)
        proportion_test = recover_lists_from_dictionnary(columns, dict_unique_values, proportion_test_concat, i)
        combi_list = np.array([a[0] for a in recover_lists_from_dictionnary(columns, dict_unique_values, combi_concat, i)]).astype(int)
        combi_names = columns[combi_list]
        
        df_scores_by_cat = get_df_scores_by_cat(proportion, proportion_test, combi_list, combi_names, i)
        
        if save:
            df_scores_by_cat.to_csv(os.path.join(dir_path_save_results,"scores_by_cat_"+ i +".csv"), sep=";")
            generate_histogram(df_scores_by_cat, dir_path_save_results, i)
            generate_plot(proportion, proportion_test, dir_path_save_results, i)
            generate_plot_plotly(proportion_concat, proportion_test_concat, combi_names_concat, values_test_concat, dir_path_save_results, i)
        df_scores_median.loc[dict_title[i]] =  df_scores_by_cat[["SRMSE","Hellinger","Pearson","R2"]].median()
        df_scores_mean.loc[dict_title[i]] = df_scores_by_cat[["SRMSE","Hellinger","Pearson","R2"]].mean()
        df_scores_agg.loc[dict_title[i]] = pd.Series(get_scores_agg(proportion_test,proportion),index=["SRMSE","Hellinger","Pearson","R2"])
        
        proportion_file = f"{basename_file_save}_{i}.npy"
        proportion = np.load(proportion_file)

    df_DCR = Distance_to_Closest_Records(df_for_evaluation, dataset_reference_training, dataset_test_df_dcr, df_info.reset_index(),proportion_test_file)

    if save:
        generate_histogram_DCR(df_DCR, dir_path_save_results)
    Wasserstein_distance_DCR = np.sqrt(1/(len(df_DCR)**2)*np.sum(np.power(df_DCR["DCR train"].sort_values().values - df_DCR["DCR test"].sort_values().values,2)))
    
    
    df_DCR_training_data = pd.read_csv(reference_DCR_file, sep=";")    
    reference_WDCR = np.sqrt(1/(len(df_DCR_training_data)**2)*np.sum(np.power(df_DCR_training_data["DCR test"].values,2)))
    
    rate_non_geo_list, rate_geo_list = [],[]
    for i in range(1,4):
        proportion_file = f"{basename_file_save}_{i}.npy"
        proportion = np.load(proportion_file)

        proportion_test = np.load(f"{proportion_test_file}_{i}.npy")
        values = np.load(f"{proportion_test_file}_{i}_values.npy", allow_pickle=True)
        combs = np.load(f"{proportion_test_file}_{i}_comb.npy")
        rate_non_geo, rate_geo = get_rate_of_impossible_combinations(df_for_evaluation,df_info, proportion_test, proportion, columns, combs, values)
        rate_non_geo_list.append(rate_non_geo)
        rate_geo_list.append(rate_geo)

    if save:
        df_scores_agg.to_csv( os.path.join(dir_path_save_results, "scores.csv"), sep=";")
        df_scores_mean.to_csv( os.path.join(dir_path_save_results, "scores_mean.csv"), sep=";")
        df_scores_median.to_csv( os.path.join(dir_path_save_results, "scores_median.csv"), sep=";")

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
    Ser_review["Rate of copies (training)"] = f'{np.abs(get_proportion_from_original_data_df(dataset_reference_training,dataset_test_df,columns)-get_proportion_from_original_data_df(dataset_reference_training,df_for_evaluation,columns)):.2%}'
    Ser_review["Rate of copies (training, without geo)"] = f'{np.abs(get_proportion_from_original_data_df(dataset_reference_training,dataset_test_df,columns_without_geo)-get_proportion_from_original_data_df(dataset_reference_training,df_for_evaluation,columns_without_geo)):.2%}'

    val = get_proportion_from_original_data_df_not_in_other_df(dataset_test_df,dataset_reference_training,df_for_evaluation,columns)
    if (val=="NA"):
        Ser_review["Rate of copies (in testing not in training)"] = 'NA'
    else:
        Ser_review["Rate of copies (in testing not in training)"] = f'{val:.2%}'
    val = get_proportion_from_original_data_df_not_in_other_df(dataset_test_df,dataset_reference_training,df_for_evaluation,columns_without_geo)
    if (val=="NA"):
        Ser_review["Rate of copies (in testing not in training, without geo)"] = 'NA'
    else:
        Ser_review["Rate of copies (in testing not in training, without geo)"] = f'{val:.2%}'
    Ser_review["Wasserstein-DCR"] = Wasserstein_distance_DCR
    Ser_review["Wasserstein-DCR ratio"] = Wasserstein_distance_DCR/reference_WDCR
    
    return Ser_review


def full_evaluation(args, term_evaluation=""):
    
    ##################
    ### Parameters ###
    ##################
        
    if not ("special_model" in args.__dict__.keys()):
        args.special_model= ""
        
    dir_path_evaluation_generated_data = get_folder_evaluation(args, term_evaluation)
    
    datapath_training = get_data_dir(args)
    datapath_testing = get_testing_data_dir(args)
    
    attr_setname = args.attributes_setname
    
    filename_training = args.filename_training
    
    if not os.path.isdir(dir_path_evaluation_generated_data):
        os.makedirs(dir_path_evaluation_generated_data)
            

    filename = get_file_path_sampling(args, term_evaluation)
    basename = get_filename_sampling(args)
    
    folder_proportion_file_generated_data =  os.path.join(dir_path_evaluation_generated_data,"Proportion_save")
    if not os.path.isdir(folder_proportion_file_generated_data):
        os.makedirs(folder_proportion_file_generated_data)
    
    file_test = os.path.join(datapath_training, args.filename_test.split(".csv")[0] + args.special_model+".csv")
    file_test_WDCR = os.path.join(datapath_testing, args.filename_test_WDCR.split(".csv")[0] + args.special_model+".csv")
    file_train = os.path.join(datapath_testing, filename_training)
    
    #################
    ### Load data ###
    #################

    df_info = get_info_file(args)

    columns = df_info["Variable_name"].to_numpy()
    name_cat = df_info["Variable_name"][(df_info["Type"].isin(["binary","boolean","category"]))].to_list()

    def load_data(filename, df_info):
        data = pd.read_csv(filename, sep=";", low_memory=False,usecols=df_info["Variable_name"])
        for idx in df_info[df_info["Type"].isin(["binary","boolean","category"])]["Variable_name"]:
            data[idx] = data[idx].astype(str)
        for idx in df_info[df_info["Type"].isin(["int"])]["Variable_name"]:
            data[idx] = data[idx].astype(int)
        for idx in df_info[df_info["Type"].isin(["float"])]["Variable_name"]:
            data[idx] = data[idx].astype(float)
        return data  


    dataset_train = load_data(file_train, df_info)

    dataset_test = load_data(file_test, df_info)
    dataset_test_WDCR = load_data(file_test_WDCR, df_info)

    min_size_category = args.transform.cat_min_count
    dataset_train,list_df  = preprocessing_cat_data_dataframe_sampling(dataset_train, min_size_category, name_cat, [dataset_test, dataset_test_WDCR])
    dataset_test = list_df[0]  
    dataset_test_WDCR = list_df[1]  
    
    
    df_sample = load_data(filename, df_info)

    dict_unique_values = {}
    for col in columns:
        folder_unique_file =  os.path.join("results","Proportion_save","Unique_Values_Census_2021",filename_training.split(".csv")[0])
        unique_file =  os.path.join(folder_unique_file,"unique_values_{col}.npy")
        if (not os.path.isfile(unique_file)):
            if (not os.path.isdir(folder_unique_file)):
                os.makedirs(folder_unique_file)
            unique_values = np.sort(dataset_test[col].unique())
            np.save(unique_file, unique_values, allow_pickle=True)
            
        unique_values = np.load(unique_file, allow_pickle=True).astype(str)
        dict_unique_values[col] = np.sort(unique_values)

    if (args.dataset_evaluation!=args.dataname): 
        # For future functionalities
        folder_test_file =  os.path.join("results","Proportion_save",attr_setname,args.dataset_evaluation,args.dataname)
        name_test_file = f"Proportion_test_data_distribution_{args.dataset_evaluation}_reference_data_{args.dataname}_{(filename_training).split(".")[0]}_{attr_setname}_{args.filename_test.split(".csv")[0]+args.special_model}"
        WDCR_reference_file = f"WDCR_{args.dataset_evaluation}_{args.dataname}"
    else:
        folder_test_file = os.path.join("results","Proportion_save",attr_setname,args.dataname,args.dataname)
        name_test_file = f"Proportion_reference_data_distribution_{args.dataname}_{(filename_training).split(".")[0]}_{attr_setname}_{args.filename_test.split(".csv")[0]+args.special_model}"
        WDCR_reference_file = f"WDCR_{args.dataname}_{args.dataname}"

    proportion_test_file = os.path.join(folder_test_file, "Proportion_reference_data_" + args.dataname + "_" + (filename_training).split(".")[0] + "_" + attr_setname + "_" + args.filename_test.split(".csv")[0] + args.special_model)

    for i in range(1,4):    
        if (not os.path.isfile(f"{proportion_test_file}_{i}.npy")):
            if(not os.path.isdir(os.path.dirname(os.path.join(folder_test_file,name_test_file)))):
                os.makedirs(os.path.dirname(os.path.join(folder_test_file, name_test_file)))
            print("\nGenerate Proportion file for test data")
            
            compute_proportion_file_from_unique_array_and_df(dict_unique_values, dataset_test.astype(str), columns, proportion_test_file, i, ".", True)
        else:
            print(f"Shape ({i}):", np.load(f"{proportion_test_file}_{i}.npy").shape)

    if (not os.path.isfile(os.path.join(folder_test_file,WDCR_reference_file,".csv"))):
        df_DCR_training_data = Distance_to_Closest_Records(dataset_train, dataset_train, dataset_test_WDCR, df_info.reset_index(),proportion_test_file)
        df_DCR_training_data.to_csv(os.path.join(folder_test_file, WDCR_reference_file, ".csv"), sep=";", index=False)

    ser_scores_generated = output_serie_eval(dict_unique_values,  os.path.join(folder_proportion_file_generated_data, basename),  os.path.join(folder_test_file, WDCR_reference_file,".csv"), proportion_test_file, df_sample, dataset_test, dataset_test_WDCR, dataset_train, df_info, dir_path_evaluation_generated_data, True)
    
    res_pandas = pd.DataFrame([ser_scores_generated], index=[f"{args.attributes_setname}_{args.size_data_str}{term_evaluation}"])    
    res_pandas.transpose().to_csv(os.path.join(dir_path_evaluation_generated_data,"overview_score.csv"), sep=";", index_label="Metric")