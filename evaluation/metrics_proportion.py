import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter


def myplot(x, y, s, bins=3000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def plot_heatmap(x,y,name):
    fig, axs = plt.subplots(2, 3)

    sigmas = [0, 16, 32, 64, 128, 256]

    for ax, s in zip(axs.flatten(), sigmas):
        if s == 0:
            ax.plot(x, y, 'k.', markersize=5)
            ax.set_title("Scatter plot")
        else:
            img, extent = myplot(x, y, s)
            ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
            ax.set_title("Smoothing with  $\sigma$ = %d" % s)
    plt.savefig(name)
    plt.close()

def SRMSE(original_data: np.ndarray, generated_data: np.ndarray,Nb: int):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    Nb: total number of bins, ie number of different combinations of attributes. In this case, Nb = len(original_data) = len(generated_data)
    '''
    return np.sqrt(np.sum(np.power(original_data-generated_data,2))/Nb)/(np.sum(original_data)/Nb)

def RMSE_per_cat(original_data: np.ndarray, generated_data: np.ndarray, combi_array: np.ndarray):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    Nb: total number of bins, ie number of different combinations of attributes. In this case, Nb = len(original_data) = len(generated_data)
    '''
    unique_combi = np.unique(combi_array,axis=0)
    scores = []
    for combi in tqdm(unique_combi):
        original_data_temp = original_data[(combi_array==combi).all(1)]
        generated_data_temp = generated_data[(combi_array==combi).all(1)]
        Nb = len(original_data_temp)
        scores.append(np.sqrt(np.sum(np.power(original_data_temp-generated_data_temp,2))/Nb))
    score = np.mean(scores)
    return score

    
def Pearson(original_data: np.ndarray, generated_data: np.ndarray):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    '''
    mu_or, mu_gen = np.mean(original_data), np.mean(generated_data)
    sig_or, sig_gen = np.std(original_data), np.std(generated_data)
    return np.mean((original_data-mu_or)*(generated_data-mu_gen))/(sig_or*sig_gen)

def R2(original_data: np.ndarray, generated_data: np.ndarray):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    '''
    mu_or = np.mean(original_data)
    return 1- np.sum(np.power(original_data-generated_data,2))/(np.sum(np.power(original_data-mu_or,2)))

def Hellinger_distance(original_data: np.ndarray, generated_data: np.ndarray):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    ''' 
    return 1/2*np.sum(np.power(np.sqrt(original_data)-np.sqrt(generated_data),2))

def Hellinger_distance_per_cat(original_data: np.ndarray, generated_data: np.ndarray, combi_array: np.ndarray):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    ''' 
    unique_combi = np.unique(combi_array,axis=0)
    scores = []
    for combi in tqdm(unique_combi):
        original_data_temp = original_data[(combi_array==combi).all(1)]
        generated_data_temp = generated_data[(combi_array==combi).all(1)]
        scores.append(1/2*np.sum(np.power(np.sqrt(original_data_temp)-np.sqrt(generated_data_temp),2)))
    score = np.mean(scores)
    return score

def distance_1_per_cat(original_data: np.ndarray, generated_data: np.ndarray, combi_array: np.ndarray):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    ''' 
    unique_combi = np.unique(combi_array,axis=0)
    scores = []
    for combi in tqdm(unique_combi):
        original_data_temp = original_data[(combi_array==combi).all(1)]
        generated_data_temp = generated_data[(combi_array==combi).all(1)]
        scores.append(np.sum(np.abs(original_data_temp-generated_data_temp)))
    score = np.mean(scores)
    return score


def is_in_data(sample: np.ndarray, data: np.ndarray):
    return (sample==data).all(1).any()

def number_of_copies_2(train_data: pd.DataFrame, generated_data: pd.DataFrame):
    res = np.zeros(len(generated_data),bool)
    pbar = tqdm(range(len(generated_data)))
    tot_true = 0
    idx = len(train_data)
    for i in pbar:
        train_data.loc[idx] = generated_data.loc[i]
        res[i] = (generated_data[i]==train_data).all(1).any()
        if (res[i]):
            tot_true+=1
        pbar.set_description(f"Current score: {np.round(tot_true/(i+1),4)}")
    return res

def number_of_copies(train_data: np.ndarray, generated_data: np.ndarray):
    res = np.zeros(len(generated_data),bool)
    pbar = tqdm(range(len(generated_data)))
    tot_true = 0
    for i in pbar:
        res[i] = (generated_data[i]==train_data).all(1).any()
        if (res[i]):
            tot_true+=1
        pbar.set_description(f"Current score: {np.round(tot_true/(i+1),4)}")
    return res

def number_of_copies_self(generated_data: np.ndarray):
    res = np.zeros(len(generated_data),bool)
    pbar = tqdm(range(len(generated_data)))
    tot_true = 0
    for i in pbar:
        res[i] = (generated_data[i]==generated_data[i+1:]).all(1).any()
        if (res[i]):
            tot_true+=1
        pbar.set_description(f"Current score: {np.round(tot_true/(i+1),4)}")
    return res

def get_measures_proportion(folder_tot,basename_tot,folder_gen,basename_gen, pretreatment_value = False, file_training = None, list_column = None):
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    df_scores = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    if (pretreatment_value):
        if (file_training is None):
            prop_generated = np.load(f"../Results/{folder_gen}/{basename_gen}_1.npy").reshape(-1)
            comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_1_comb.npy").reshape(-1)
            value_comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_1_values.npy", allow_pickle=True).reshape(-1)
            idx_need_treatment = np.unique(comb_tot[prop_generated==0])
            res = []
            for idx in idx_need_treatment:
                res.append(value_comb_tot[(prop_generated==0)&(comb_tot==idx)])
            print(res)
        else:
            training_data = np.load(f"../Results/{folder_tot}/{file_training}.npy", allow_pickle=True).T
            comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_1_comb.npy").reshape(-1)
            value_comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_1_values.npy", allow_pickle=True).reshape(-1)
            values_tot = get_values(comb_tot, value_comb_tot)
            val_training_data = []
            for i in range(len(training_data)):
                val_training_data.append(np.unique(training_data[i]))
            idx_need_treatment = []
            res = []
            for i in range(len(values_tot)):
                if not(same_values(values_tot[i],val_training_data[i])):
                    idx_need_treatment.append(i)
                    res.append(np.array(list(set(values_tot[i]).symmetric_difference(set(val_training_data[i])))))
            idx_need_treatment_keep = []
            for i in range(len(idx_need_treatment)):
                if(res[i].dtype!=np.int32):
                    idx_need_treatment_keep.append(i)
            idx_need_treatment = [idx_need_treatment[i] for i in idx_need_treatment_keep]
            res = [res[i] for i in idx_need_treatment_keep]
            print(res,idx_need_treatment)
            print(comb_tot)
    for i in range(1,4):
        prop_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}.npy")
        prop_generated = np.load(f"../Results/{folder_gen}/{basename_gen}_{i}.npy")
        if (pretreatment_value):
            comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}_comb.npy").astype(int)
            value_comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}_values.npy", allow_pickle=True)
            for values, column in zip(res,idx_need_treatment):
                prop_tot = recompute_scores(prop_tot, comb_tot, value_comb_tot, values, column)
        print("____________________________")
        print(i)

        print("SRMSE",SRMSE(prop_tot,prop_generated,len(prop_tot)))
        print("Hellinger",Hellinger_distance(prop_tot,prop_generated))
        print("R2",R2(prop_tot,prop_generated))
        print("Pearson",Pearson(prop_tot,prop_generated))
        plt.figure(figsize=[7,7])
        plt.plot([-1,1], [-1,1], color='r', alpha=0.5)
        plt.scatter(prop_tot,prop_generated)
        plt.xlim(-0.02,np.max([prop_tot,prop_generated])+0.02)
        plt.ylim(-0.02,np.max([prop_tot,prop_generated])+0.02)
        plt.xlabel("Original data proportion")
        plt.ylabel("Sampled data proportion")
        plt.title(dict_title[i])
        plt.savefig(f"../Results/{folder_gen}/comparison_{i}.png")
        plt.close()
        plot_heatmap(prop_tot,prop_generated,f"../Results/{folder_gen}/comparison_heatmap_{i}.png")
    print(df_scores.transpose().to_markdown())
    df_scores.to_csv(f"../Results/{folder_gen}/scores.csv", sep=";")

def get_plotly_plot(folder_tot,basename_tot,folder_gen,basename_gen, pretreatment_value = False, file_training = None, list_column = None):
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    df_scores = pd.DataFrame(columns=["SRMSE","Pearson","R2"])
    if (pretreatment_value):
        if (file_training is None):
            prop_generated = np.load(f"../Results/{folder_gen}/{basename_gen}_1.npy").reshape(-1)
            comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_1_comb.npy").reshape(-1)
            value_comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_1_values.npy", allow_pickle=True).reshape(-1)
            idx_need_treatment = np.unique(comb_tot[prop_generated==0])
            res = []
            for idx in idx_need_treatment:
                res.append(value_comb_tot[(prop_generated==0)&(comb_tot==idx)])
        else:
            training_data = np.load(f"../Results/{folder_tot}/{file_training}.npy", allow_pickle=True).T
            comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_1_comb.npy").reshape(-1)
            value_comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_1_values.npy", allow_pickle=True).reshape(-1)
            values_tot = get_values(comb_tot, value_comb_tot)
            val_training_data = []
            for i in range(len(training_data)):
                val_training_data.append(np.unique(training_data[i]))
            idx_need_treatment = []
            res = []
            for i in range(len(values_tot)):
                if not(same_values(values_tot[i],val_training_data[i])):
                    idx_need_treatment.append(i)
                    res.append(np.array(list(set(values_tot[i]).symmetric_difference(set(val_training_data[i])))))
            idx_need_treatment_keep = []
            for i in range(len(idx_need_treatment)):
                if(res[i].dtype!=np.int32):
                    idx_need_treatment_keep.append(i)
            idx_need_treatment = [idx_need_treatment[i] for i in idx_need_treatment_keep]
            res = [res[i] for i in idx_need_treatment_keep]
    for i in range(1,4):
        prop_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}.npy")
        prop_generated = np.load(f"../Results/{folder_gen}/{basename_gen}_{i}.npy")
        comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}_comb.npy").astype(int)
        value_comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}_values.npy", allow_pickle=True)
        if (pretreatment_value):
            for values, column in zip(res,idx_need_treatment):
                prop_tot = recompute_scores(prop_tot, comb_tot, value_comb_tot, values, column)
        print("____________________________")
        print(i)
        df_scores.loc[dict_title[i]] = [SRMSE(prop_tot,prop_generated,len(prop_generated)), Pearson(prop_tot,prop_generated), R2(prop_tot,prop_generated)] 
        if(not (list_column is None)):
            dict_col = {i: list_column[i] for i in range(len(list_column))}
            comb_tot = np.vectorize(dict_col.get)(comb_tot)
        if(len(comb_tot)<10000):
            fig = go.Figure(data = go.Scatter(x=prop_tot, y = prop_generated,
                                            # meta={"combi": comb_tot, "value": value_comb_tot},
                                            mode="markers",
                                            customdata=np.stack([comb_tot, value_comb_tot],axis=1),
                                            hovertemplate="Proportion original: %{x}<br>Proportion generated: %{y}<br><br>Combination: %{customdata[0]}<br>Value: %{customdata[1]}"))
            fig.write_html(f"../Results/{folder_gen}/comparison_plotly_{i}.html")
    print(df_scores.transpose().to_markdown())
    df_scores.to_csv(f"../Results/{folder_gen}/scores.csv", sep=";")

def get_df_scores_by_cat(prop_sample, prop_original, combi, i):
    df = pd.DataFrame()
    for j in range(i):
        df[f'idx_{j}'] = combi[:,j]
    df['prop_generated'] = prop_sample
    df['prop_original'] = prop_original
    df['prop_generated_2'] = np.power(prop_sample,2)
    df['prop_original_2'] = np.power(prop_original,2)
    df['prop_original_generated'] = prop_original*prop_sample
    df["err_2"] = np.power(prop_sample-prop_original,2)
    df["Hellinger"] = 1/2*np.power(np.sqrt(prop_sample)-np.sqrt(prop_original),2)
    df["count"] = 1
    df_temp = df.groupby([f'idx_{j}' for j in range(i)]).sum()
    var = df.groupby([f'idx_{j}' for j in range(i)])["prop_original"].var()
    df_temp["SRMSE"] = np.sqrt(df_temp["err_2"]*df_temp["count"])
    df_temp["Hellinger"] = df_temp["Hellinger"]
    df_temp["Pearson"] = (df_temp["count"]*df_temp["prop_original_generated"]-df_temp["prop_generated"]*df_temp["prop_original"])/(np.sqrt(df_temp["count"]*df_temp["prop_original_2"]-df_temp["prop_original"]*df_temp["prop_original"])*np.sqrt(df_temp["count"]*df_temp["prop_generated_2"]-df_temp["prop_generated"]*df_temp["prop_generated"]))
    df_temp["R2"] = 1-df_temp["err_2"]/(var*(df_temp["count"]-1))
    return df_temp

def get_scores_by_cat(prop_sample, prop_original, combi, i):
    df = pd.DataFrame()
    for j in range(i):
        df[f'idx_{j}'] = combi[:,j]
    df['prop_generated'] = prop_sample
    df['prop_original'] = prop_original
    df['prop_generated_2'] = np.power(prop_sample,2)
    df['prop_original_2'] = np.power(prop_original,2)
    df['prop_original_generated'] = prop_original*prop_sample
    df["err_2"] = np.power(prop_sample-prop_original,2)
    df["Hellinger"] = 1/2*np.power(np.sqrt(prop_sample)-np.sqrt(prop_original),2)
    df["count"] = 1
    df_temp = df.groupby([f'idx_{j}' for j in range(i)]).sum()
    var = df.groupby([f'idx_{j}' for j in range(i)])["prop_original"].var()
    df_temp["SRMSE"] = np.sqrt(df_temp["err_2"]*df_temp["count"])
    df_temp["Hellinger"] = df_temp["Hellinger"]
    df_temp["Pearson"] = (df_temp["count"]*df_temp["prop_original_generated"]-df_temp["prop_generated"]*df_temp["prop_original"])/(np.sqrt(df_temp["count"]*df_temp["prop_original_2"]-df_temp["prop_original"]*df_temp["prop_original"])*np.sqrt(df_temp["count"]*df_temp["prop_generated_2"]-df_temp["prop_generated"]*df_temp["prop_generated"]))
    df_temp["R2"] = 1-df_temp["err_2"]/(var*(df_temp["count"]-1))
    return df_temp[["SRMSE","Hellinger","Pearson","R2"]].median(),df_temp[["SRMSE","Hellinger","Pearson","R2"]].mean() # or mean

def get_scores_agg(prop_tot,prop_generated):
    SRMSE_score = SRMSE(prop_tot,prop_generated,len(prop_generated))
    Hellinger_score = Hellinger_distance(prop_tot,prop_generated)
    Pearson_score = Pearson(prop_tot,prop_generated)
    R2_score = R2(prop_tot,prop_generated)
    return SRMSE_score, Hellinger_score, Pearson_score, R2_score


def mean_SRMSE_by_category(combi,prop_sample, prop_original,i):
    df = pd.DataFrame()
    for j in range(i):
        df[f'idx_{j}'] = combi[:,j]
    df["err_2"] = np.power(prop_sample-prop_original,2)
    df["count"] = 1
    df_temp = df.groupby([f'idx_{j}' for j in range(i)]).sum()
    df_temp["SRMSE"] = np.sqrt(df_temp["err_2"]*df_temp["count"])
    return df_temp["SRMSE"].mean()

def temp_scores(folder_tot,basename_tot,folder_gen,basename_gen):
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    df_scores_median = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    df_scores_mean = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    df_scores_agg = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    for i in range(1,4):
        prop_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}.npy")
        prop_generated = np.load(f"../Results/{folder_gen}/{basename_gen}_{i}.npy")
        comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}_comb.npy").astype(int)
        median_scores, mean_scores = get_scores_by_cat(prop_generated,prop_tot,comb_tot,i)

        df_scores_median.loc[dict_title[i]] = median_scores
        df_scores_mean.loc[dict_title[i]] = mean_scores
        df_scores_agg.loc[dict_title[i]] = pd.Series(get_scores_agg(prop_tot,prop_generated),index=["SRMSE","Hellinger","Pearson","R2"])
    print("### Aggregate scores\n",df_scores_agg.transpose().to_markdown())
    print("### Median scores\n",df_scores_median.transpose().to_markdown())
    print("### Mean scores\n",df_scores_mean.transpose().to_markdown())
    df_scores_agg.to_csv(f"../Results/{folder_gen}/scores.csv", sep=";")
    df_scores_mean.to_csv(f"../Results/{folder_gen}/scores_mean.csv", sep=";")
    df_scores_median.to_csv(f"../Results/{folder_gen}/scores_median.csv", sep=";")



def get_values(comb,values):
    res = []
    for c in np.unique(comb):
        val_comb = values[comb==c].copy()
        res.append(val_comb)
    return res

def same_values(val1,val2):
    if (len(val1)!=len(val2)):
        return False
    for val in val1:
        if not ((val==val2).any()):
            return False
    return True

def recompute_scores(proportion_original, comb_original, value_original, values_to_delete, column):
    idx_comb = (column==comb_original).any(1)
    comb_unique = np.unique(comb_original[idx_comb],axis=0)
    values_to_delete = np.expand_dims(values_to_delete,1)
    for comb in tqdm(comb_unique):
        idx_comb = (comb==comb_original).all(1)
        id_col = np.argwhere(comb==column)[0][0]
        value_original_temp = value_original[idx_comb]
        idx_to_zeros = (value_original_temp[:,id_col]==values_to_delete).any(0)
        lamb = 1-np.sum(proportion_original[idx_comb][idx_to_zeros])
        proportion_original[np.arange(len(idx_comb),dtype=int)[idx_comb][idx_to_zeros]] = 0
        proportion_original[idx_comb] = proportion_original[idx_comb]/lamb
    return proportion_original