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

def SRMSE_aggregated_scores(original_data: np.ndarray, generated_data: np.ndarray,Nb: int):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    Nb: total number of bins, ie number of different combinations of attributes. In this case, Nb = len(original_data) = len(generated_data)
    '''
    return np.sqrt(np.sum(np.power(original_data-generated_data,2))/Nb)/(np.sum(original_data)/Nb)


def Pearson_aggregated_scores(original_data: np.ndarray, generated_data: np.ndarray):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    '''
    mu_or, mu_gen = np.mean(original_data), np.mean(generated_data)
    sig_or, sig_gen = np.std(original_data), np.std(generated_data)
    return np.mean((original_data-mu_or)*(generated_data-mu_gen))/(sig_or*sig_gen)

def R2_aggregated_scores(original_data: np.ndarray, generated_data: np.ndarray):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    '''
    mu_or = np.mean(original_data)
    return 1- np.sum(np.power(original_data-generated_data,2))/(np.sum(np.power(original_data-mu_or,2)))

def Hellinger_distance_aggregated_scores(original_data: np.ndarray, generated_data: np.ndarray):
    '''
    original_data: propostion of each combination for the original data
    generated_data: propostion of each combination for the generated data in the same order that the original data
    ''' 
    return 1/2*np.sum(np.power(np.sqrt(original_data)-np.sqrt(generated_data),2))

def compute_SRMSE_from_freq_list(freq_list_1, freq_list_2):
    res = []
    for (freq_1,freq_2) in zip(freq_list_1, freq_list_2):
        SRMSE = np.sqrt(np.sum(np.power(freq_1-freq_2,2))*len(freq_1))
        res.append(SRMSE)
    return np.mean(res)



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


def get_plotly_plot(folder_tot,basename_tot,folder_gen,basename_gen, pretreatment_value = False, file_training = None, list_column = None):
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    df_scores = pd.DataFrame(columns=["SRMSE","Pearson","R2"])
    for i in range(1,4):
        prop_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}.npy")
        prop_generated = np.load(f"../Results/{folder_gen}/{basename_gen}_{i}.npy")
        comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}_comb.npy").astype(int)
        value_comb_tot = np.load(f"../Results/{folder_tot}/{basename_tot}_{i}_values.npy", allow_pickle=True)
        print("____________________________")
        print(i)
        df_scores.loc[dict_title[i]] = [SRMSE_aggregated_scores(prop_tot,prop_generated,len(prop_generated)), Pearson_aggregated_scores(prop_tot,prop_generated), R2_aggregated_scores(prop_tot,prop_generated)] 
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

def get_df_scores_by_cat(props_sample, props_original, combi, combi_names, i):
    df = pd.DataFrame()
    len_props = [len(prop_sample) for prop_sample in props_sample]
    for j in range(i):
        df[f'idx_{j}'] = combi[:,j]
        df[f'Attributes_{j}'] = combi_names[:,j]
    df["Len"] = len_props
    df["SRMSE"] = [np.sqrt(np.sum(np.power(prop_sample-prop_original,2))*len(prop_sample)) for prop_sample, prop_original in zip(props_sample, props_original)]
    df["Hellinger"] = [np.sum(1/2*np.power(np.sqrt(prop_sample)-np.sqrt(prop_original),2))  for prop_sample, prop_original in zip(props_sample, props_original)]
    df["Pearson"] = [(np.sum((prop_original-np.mean(prop_original))*(prop_sample-np.mean(prop_sample))))/(np.sqrt(np.sum(np.power(prop_original-np.mean(prop_original),2))*np.sum(np.power(prop_sample-np.mean(prop_sample),2)))) for prop_sample, prop_original in zip(props_sample, props_original)] 
    df["R2"] = [1-(np.sum(np.power(prop_sample-prop_original,2))/(np.var(prop_original)*(len(prop_sample)-1))) for prop_sample, prop_original in zip(props_sample, props_original)]
    return df

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
    # df_temp = df
    df_temp["SRMSE"] = np.sqrt(df_temp["err_2"]*df_temp["count"])
    df_temp["Hellinger"] = df_temp["Hellinger"]#/df_temp["count"]
    df_temp["Pearson"] = (df_temp["count"]*df_temp["prop_original_generated"]-df_temp["prop_generated"]*df_temp["prop_original"])/(np.sqrt(df_temp["count"]*df_temp["prop_original_2"]-df_temp["prop_original"]*df_temp["prop_original"])*np.sqrt(df_temp["count"]*df_temp["prop_generated_2"]-df_temp["prop_generated"]*df_temp["prop_generated"]))
    df_temp["R2"] = 1-df_temp["err_2"]/(var*(df_temp["count"]-1))
    return df_temp[["SRMSE","Hellinger","Pearson","R2"]].median(),df_temp[["SRMSE","Hellinger","Pearson","R2"]].mean() # or mean

def get_scores_agg(prop_tot,prop_generated):
    prop_tot_aggregated = np.concat(prop_tot)
    prop_generated_aggregated = np.concat(prop_generated)
    SRMSE_score = SRMSE_aggregated_scores(prop_tot_aggregated,prop_generated_aggregated,len(prop_generated_aggregated))
    Hellinger_score = Hellinger_distance_aggregated_scores(prop_tot_aggregated,prop_generated_aggregated)
    Pearson_score = Pearson_aggregated_scores(prop_tot_aggregated,prop_generated_aggregated)
    R2_score = R2_aggregated_scores(prop_tot_aggregated,prop_generated_aggregated)
    return SRMSE_score, Hellinger_score, Pearson_score, R2_score
