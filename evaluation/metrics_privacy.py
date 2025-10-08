import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def generate_histogram_DCR(df, path):
    """Generate and save a histogram of DCR values."""
    plt.figure(figsize=[7,7])
    for set in df.columns:
        quantile = df[set].quantile(0.99)  #avoid long tail
        values = df[set][df[set]<=quantile]
        if ((values.max()-values.min())<(np.quantile(df.to_numpy(),0.99)-np.quantile(df.to_numpy(),0.01))*0.3): #avoid invisible plot
            id_min = np.arange(len(df))[df[set]==df[set].min()]
            id_max = np.arange(len(df))[df[set]==df[set].max()]
            values.iloc[id_min[0]] = np.quantile(df.to_numpy(),0.01)
            values.iloc[id_max[-1]] = np.quantile(df.to_numpy(),0.99)
        plt.hist(values,100,label=set)
    plt.title(f'Distribution DCR')
    plt.legend()
    plt.savefig(f"{path}/hist_DCR.png")
    plt.close()

def load_proportion_data(proportion_test_file):
    """Load proportion-related data from .npy files."""
    proportion_test = np.load(f"{proportion_test_file}_1.npy").reshape(-1)
    combi_test = np.load(f"{proportion_test_file}_1_comb.npy").reshape(-1)
    value_test = np.load(f"{proportion_test_file}_1_values.npy", allow_pickle=True).reshape(-1)
    return proportion_test, combi_test, value_test

def preprocess_numerical_data(value_test, combi_test, proportion_test, col_num):
    """Compute the cumulative probabilities for numerical columns."""
    repartition_function_num = []
    values_num = []
    
    for id in col_num.index:
        values_variable = value_test[combi_test == id]
        order = np.argsort(values_variable)
        cum_prob = np.cumsum(proportion_test[combi_test == id][order])
        temp = np.concatenate([[0], cum_prob])
        centered_cum_prob = (temp[:-1] + temp[1:]) / 2
        
        repartition_function_num.append(centered_cum_prob)
        values_num.append(values_variable[order])
    
    return repartition_function_num, values_num

def transform_continuous(values, repartition_function, v):
    """Transform continuous values using the cumulative distribution."""
    sup_v = np.min(np.concatenate([np.arange(len(values))[values > v], [len(values) - 1]]))
    inf_v = np.max(np.concatenate([np.arange(len(values))[values <= v],[0]]))
    repartition_function_v = repartition_function[inf_v] + \
        (v - inf_v) / (sup_v - inf_v + 1e-3) * (repartition_function[sup_v] - repartition_function[inf_v])
    return repartition_function_v

def transform_categorical_data(df, df_ref, col_cat):
    """Transform categorical data using OneHotEncoder."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(df_ref[col_cat].astype(str))
    encoded_data = encoder.transform(df[col_cat].astype(str))
    return encoded_data/np.sqrt(2)


def transform_numerical_data(df, col_num, values_num, repartition_function_num):
    """Apply transformations to numerical columns in the DataFrame."""
    df_transformed = df.copy()
    for i, col in enumerate(col_num):
        df_transformed[col] = df[col].apply(
            lambda v: transform_continuous(values_num[i], repartition_function_num[i], v)
        )
    return df_transformed

def Distance_to_Closest_Records(generated_data: pd.DataFrame, training_data: pd.DataFrame, test_data: pd.DataFrame, df_info: pd.DataFrame, proportion_test_file: str):
    print("Start Computation of Distance to Closest Records")
    # Load proportion-related data
    proportion_test, combi_test, value_test = load_proportion_data(proportion_test_file)

    # Identify categorical and numerical columns
    col_cat = df_info[df_info["Type"].isin(["binary", "cat", "bool"])]["Variable_name"]
    col_num = df_info[df_info["Type"].isin(["cont", "int"])]["Variable_name"]
    
    # Transform numerical data
    repartition_function_num, values_num = preprocess_numerical_data(value_test, combi_test, proportion_test, col_num)

    # Ensure matching sizes between training and test data
    print(f"Datasets size:\tTraining:{len(training_data)}\tTesting:{len(test_data)}\tGenerated:{len(generated_data)}")
    min_len = min(len(training_data), len(test_data))
    training_data = training_data.head(min_len)
    test_data_original = test_data.copy()
    test_data = test_data.head(min_len)
    print(f"Datasets size after croping:\tTraining:{len(training_data)}\tTesting:{len(test_data)}\tGenerated:{len(generated_data)}")

    # Apply transformations to numerical data
    generated_data_num = transform_numerical_data(generated_data[col_num], col_num, values_num, repartition_function_num)
    training_data_num = transform_numerical_data(training_data[col_num], col_num, values_num, repartition_function_num)
    test_data_num = transform_numerical_data(test_data[col_num], col_num, values_num, repartition_function_num)
    
    # Extract numerical data from the original DataFrames
    generated_data_num = generated_data_num[col_num].to_numpy()
    training_data_num = training_data_num[col_num].to_numpy()
    test_data_num = test_data_num[col_num].to_numpy()

    # Transform categorical data using OneHotEncoder
    generated_data_cat = transform_categorical_data(generated_data, test_data_original, col_cat)
    training_data_cat = transform_categorical_data(training_data, test_data_original, col_cat)
    test_data_cat = transform_categorical_data(test_data, test_data_original, col_cat)

    # Concatenate numerical and categorical data
    generated_data_transformed = np.concatenate([generated_data_num, generated_data_cat], axis=1)
    training_data_transformed = np.concatenate([training_data_num, training_data_cat], axis=1)
    test_data_transformed = np.concatenate([test_data_num, test_data_cat], axis=1)

    # Nearest Neighbors calculation
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    
    # Fit model for training data and compute distances for generated data
    nn.fit(training_data_transformed)
    dcr_train, _ = nn.kneighbors(generated_data_transformed)
    
    # Fit model for test data and compute distances for generated data
    nn.fit(test_data_transformed)
    dcr_test, _ = nn.kneighbors(generated_data_transformed)

    # Store results in a DataFrame
    res = pd.DataFrame({
        "DCR train": dcr_train.flatten(),
        "DCR test": dcr_test.flatten()
    }, index=generated_data.index)
    print("Computation of Distance to Closest Records over")
    return res