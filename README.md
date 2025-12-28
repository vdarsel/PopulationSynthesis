# Population Synthesis 

This Git deposit contains an open-source for population synthesis, proposing various probabilistic models and Deep Generative Models (DGMs). This repo aims for reproductible research and efficient model comparison. Preprocessed data from Île-de-France (region around Paris) is available for direct comparison.



# Main reference

This Git deposit is originally the source code from:
> Darsel, V., Come, E., \& Oukhellou, L. (2025) [Robust and Reproducible Evaluation Framework for Population Synthesis Models—Application to Probabilistic and Deep Generative Models.](https://dx.doi.org/10.2139/ssrn.5295092), _Pre-Print available at SSRN 5295092_.

# Tutorial

## Code pulling

Clone the Git repository on your working place with:

```bash
git clone https://github.com/vdarsel/PopulationSynthesis.git 
```

## Data preparation

Download the file ``info.csv``, and either ``datasets_Individual_0_03.zip`` or ``datasets_Individual_1.zip``. In the former, the training dataset contains 0.03% of the total population, and 1% in the second. Move the dowloaded files to the data folder (Data>French_Census_Data_2021), and unzip the zip file.

If you want to use your own dataset, more explanations on the format are given in the [Data Description](https://github.com/vdarsel/PopulationSynthesis/tree/main/data).

## Install required libraries

Create an adequate environment
```bash
conda create --name PopSynth python=3.12.3
```


Install the libraries in the environment with:
```bash
conda activate PopSynth
pip install -r requirements.txt
```
Deep Generative Models can be trained on 
If your installation is compatible, it is highly recommended to force the installation another torch version compatible with CUDA to accelerate the computation time. The link is available [here](). For instance, with Windows and Cuda 12.6:

```bash
pip install  --force-reinstall torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu126
```


## Run the experiments

The following command launches the experiments (training + sampling + evaluation).

```bash
python full_process.py --variable NAME --size_data SIZE --MODELS (--MODELS_PARAMETERS) (--no_training) (--no_evaluation)
```

|Argument           |Description                                                          |Possible Values      |
|-------------------|---------------------------------------------------------------------|---------------------|
|```--variable NAME```    |Defines the set of attributes for the experiment.                    |```basic```, ```socio```, or ```extended```                |
|```--size_data SIZE ```  |Specifies the dataset size.                                          |```0_03``` or ```1```                 |
|```--MODELS```           |Selects the model(s) to test. Multiple models can be run in sequence.|See [Models](#models)|
|```--MODELS_PARAMETERS```|Optional parameters for the selected models.                         |(Optional flag) Model-specific       |
|```--force_training```      |Skips the training phase for DGMs, as their parameters are saved after training in the ```ckpt``` folder.|(Optional flag)      |
|```--force_training_embedding```      |Skips the training phase for DGMs, as their parameters are saved after training in the ```ckpt``` folder.|(Optional flag)      |
|```--no_sampling```    |Skips the sampling phase (Warning: could lead to errors in evaluation if no data available).                                          |(Optional flag)      |
|```--no_evaluation```    |Skips the evaluation phase.                                          |(Optional flag)      |

The ```NAME``` argument corresponds to three sets of attributes:
For more details, refer to the [Data scenarios](https://github.com/vdarsel/PopulationSynthesis/tree/main/data?tab=readme-ov-file#data-scenarios) documentation.

### Example Usage
To run an experiment with the socio dataset, size 1, using the Bayesian Network Hill-Climbing model, the MCMC Frequentist approach, and the Diffusion model with reference size of 100:
```bash
python full_process.py --variable socio --size_data 1 --BN_hill --MCMC_freq --Diffusion --Diffusion_size 100
```


## Analyze the results (Output Structure)

In the default configuration, the generated population is saved in ```Results>Generated_data>French_Census_Data_2021_NAME_set_SIZE%_MODEL```, and the scores are saved in ```Results>Generated_data>French_Census_Data_2021_NAME_set_SIZE%_MODEL>French_Census_Data_2021>scores.csv```. The population is saved in a file named ```TRAINING-FILE_NAME_N_MODEL```, where ```TRAINING-FILE_NAME``` is the name of the training file, and ```N``` is thesize of the generated population. 

For a quick model comparison following the metrics proposed in [Robust and Reproducible Evaluation Framework for Population Synthesis Models—Application to Probabilistic and Deep Generative Models.](https://dx.doi.org/10.2139/ssrn.5295092), the file ```Results>Generated_data>quick_comparison.csv``` summaries the scores for the 3 metrics: $\overline{SRMSE}_3$; $SSCIOD$; and $WDCR$.

# Models

## Probabilistic Models
Two probabilistic models are implemented, each of them with two approaches.

### Bayesian Networks
Bayesian Networks are implemented with two algorithms for generating the Directed Acyclic Graph (DAG):

|Model              |Description                                                          |Flag                 | Reference        |
|-------------------|---------------------------------------------------------------------|---------------------|-----------------|
|Hill-Climbing      |Uses the BIC score to search for the optimal DAG, with hill-climbing.                    |```--BN_hill```     |  Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009 Section 18.4.3 (page 811ff)               |
|Tree Search        |Constructs a maximum-weight spanning tree using mutual information as edge weights.|```--BN_tree```           | Chow, C. K.; Liu, C.N. (1968), “Approximating discrete probability distributions with dependence trees”, IEEE Transactions on Information Theory, IT-14 (3): 462–467                |

### Monte Carlo Markov Chain

MCMC uses Gibbs Sampling to generate synthetic populations. Two approaches are available for computing conditional probabilities:
|Model              |Description                                                          |Flag                 | Options                 |
|-------------------|---------------------------------------------------------------------|---------------------|-----------------|
|Frequentist        |Transition probabilities are computed from observed frequencies.     |```--MCMC_freq```          | None |
|Bayesian           |Uses the Bayesian Posterior of a Dirichlet distribution for transition probabilities.|```--MCMC_Baysian```      | ```--MCMC_Baysian_alpha``` for the Dirichlet prior  (default=0.1) |


## Deep Generative Models

The architectures of the different models are given in the article [Robust and Reproducible Evaluation Framework for Population Synthesis Models—Application to Probabilistic and Deep Generative Models.](https://dx.doi.org/10.2139/ssrn.5295092). In the different tables, references with full model description are given with the models.

### Variationnal Autoencoder (VAE)

|Model              |Flag                 | Options                 | Reference |
|-------------------|---------------------------------------------------------------------|---------------------|-------------------------|
|$\beta$-VAE        |```--beta_VAE```          |```--beta_VAE_dim``` for the layer reference dimension (default=2000) <br>```--beta_VAE_beta``` for the $\beta$ value (default =1) |[$\beta$-VAE](https://openreview.net/forum?id=Sy2fzU9gl)|
|$\beta$-VAE with continuous pre-embedding        |```--beta_VAE_embedded```      |```--beta_VAE_embedded_dim``` for the layer reference  dimension (default=2000) <br>```--beta_VAE_embedded_beta``` for the $\beta$ value (default =1) |[$\beta$-VAE](https://openreview.net/forum?id=Sy2fzU9gl) <br> [Embedding](https://openreview.net/forum?id=4Ay23yeuz0) |
<!-- |TVAE        |```--TVAE```      |```--TVAE_dim``` for the layer reference  dimension (default=2000) <br>```--TVAE_beta``` for the $\beta$ value (default =1) | [TVAE](https://proceedings.neurips.cc/paper_files/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Paper.pdf)| -->


### Generative Adverserial Network (GAN)

|Model              |Flag                 | Options                 | Reference |
|-------------------|---------------------------------------------------------------------|---------------------|-------------------------|
|WGAN    |```--WGAN```          |```--WGAN_dim``` for the layer reference dimension (default=2000) |[WGAN](https://proceedings.mlr.press/v70/arjovsky17a.html)|
|WGAN with continuous pre-embedding        |```--WGAN_embedded```      |```--WGAN_embedded_dim``` for the layer reference dimension (default=2000) |[WGAN](https://proceedings.mlr.press/v70/arjovsky17a.html) <br> [Embedding](https://openreview.net/forum?id=4Ay23yeuz0) |



### Diffusion models
|Model              |Flag                 | Options                 | Reference |
|-------------------|---------------------------------------------------------------------|---------------------|-------------------------|
|TabSyn    |```--Diffusion```          |```--Diffusion_dim``` for the layer reference dimension (default=2000) |[TabSyn](https://openreview.net/forum?id=4Ay23yeuz0)|


# Notes

- The first time you perform an evaluation with a new test set, the evaluation takes longer, as all frequencies for the testing set are computed (and then saved).
- All parameters can be tuned in the configuration files located in the ```config``` folder. Attribute-specific parameters are defined in ```config/config_variable/NAME.yml```. Training set size-dependent parameters are specified in ```config/config_size/SIZE.yml```.
- It is not recommended to use the IRIS attribute at 0.03%, as there are 2,256 unique IRIS in the 0.03% training dataset for 3,677 individuals. The TRIRIS is more recommended (1180 modalities)


# To be added soon
- Direct Inflating (already implemented, documentation required)
- IPF
- TVAE