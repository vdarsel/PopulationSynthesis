import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def generate_histogram_DCR(df, path):
    """Generate and save a histogram of DCR values."""
    plt.figure(figsize=[7,7])
    for set in df.columns:
        quantile = df[set].quantile(0.99)  #avoid long tail
        values = df[set][df[set]<=quantile]
        if ((values.max()-values.min())<(np.quantile(df.to_numpy(),0.99)-np.quantile(df.to_numpy(),0.01))*0.3): #avoid invisible plot
            id_min = np.arange(len(values))[values==values.min()]
            id_max = np.arange(len(values))[values==values.max()]
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
        mask = (combi_test == id)
        values_variable = value_test[mask].astype(float)
        order = np.argsort(values_variable)
        cum_prob = np.cumsum(proportion_test[mask][order])
        temp = np.concatenate([[0.0], cum_prob])
        centered_cum_prob = (temp[:-1] + temp[1:]) / 2.0
        
        repartition_function_num.append(centered_cum_prob)
        values_num.append(values_variable[order])
    
    return repartition_function_num, values_num


def _transform_continuous(x: np.ndarray, values: np.ndarray, cdf: np.ndarray,
                          interpolation: str) -> np.ndarray:
    """Transform continuous values using the cumulative distribution. Vectorized function

    interpolation = "linear": proper piecewise-linear interpolation of the CDF between
                              the observed values (clamped outside their range)
    interpolation = "legacy": the exact formula of transform_continuous, in which the
                              interpolation weight is (v - inf_v) / (sup_v - inf_v + 1e-3)
                              with inf_v, sup_v the *positions* (not the values) in `values`
    """
    pos = np.searchsorted(values, x, side="right")   # number of observed values <= x
    inf = np.maximum(pos - 1, 0)                      # last index with value <= x (0 if none)
    sup = np.minimum(pos, len(values) - 1)            # first index with value > x (last if none)
    if interpolation == "legacy":
        w = (x - inf) / (sup - inf + 1e-3)
    else:
        lo, hi = values[inf], values[sup]
        denom = hi - lo
        w = np.divide(x - lo, denom, out=np.zeros_like(x), where=denom > 0)
        w = np.clip(w, 0.0, 1.0)
    out = cdf[inf] + w * (cdf[sup] - cdf[inf])
    return np.where(np.isnan(x), 0.5, out)


def _encode(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    cdfs: list[np.ndarray],
    values_num: list[np.ndarray],
    num_cols: list[str],
    cat_cols: list[str],
    cat_ref: pd.DataFrame,
    interpolation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode (generated, train, test) as float matrices; Euclidean distance on them is d."""
    parts: list[list[np.ndarray]] = [[], [], []]

    for col, values, cdf in zip(num_cols, values_num, cdfs):
        for k, df in enumerate(frames):
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            parts[k].append(_transform_continuous(x, values, cdf, interpolation)[:, None])

    for c in cat_cols:
        cats = pd.Index(cat_ref[c].astype(str).unique())
        for k, df in enumerate(frames):
            codes = pd.Categorical(df[c].astype(str), categories=cats).codes  # -1 = unknown
            onehot = np.zeros((len(df), len(cats)))
            known = codes >= 0
            onehot[np.flatnonzero(known), codes[known]] = 1.0
            parts[k].append(onehot / np.sqrt(2.0))

    return tuple(np.hstack(p) if p else np.zeros((len(f), 0)) for p, f in zip(parts, frames))

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

    
    (generated_data_transformed, training_data_transformed, test_data_transformed) = _encode((generated_data, training_data, test_data),
                                                                                     repartition_function_num, values_num,
                                                                                     col_num, col_cat, test_data_original, "linear")

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


