# Data Presentation

The code is with a publicly and full-scale dataset, that can be download here: [Data](https://data.mendeley.com/datasets/p2gcy7x7sd/). The code is designed to generate a population at the **individual-level**. To run the code, download the datasets, extract the files and copy the extracted files (not the folders) to this repository. 

The code is designed to handle data at the **individual-level** in the format of [Data](https://data.mendeley.com/datasets/p2gcy7x7sd/)., and put in the Data folder.


## Training data size

The size of the available data varies depending on the data providers. 
In the published data, two data scenarios are presented that differ by the size of the training dataset:

- At 0.03% of the total population, this scenario mimics cases where the amount of data is minimal (Household Travel Survey at the national level).
- At 1% of the total population, this scenario corresponds to a Household Travel Survey at a narrower geographical scale, or Census data thourgh Public Use Microdata Sample (PUMS).


## Attributes Scenarios


Different data scenarios are provided to test different model complexities. 

| Category               | Variable name       | Type        | # of modalities | Basic | Socio | Extended |
|------------------------|---------------------|-------------|-----------------|-------|-------|------|
| **Person attributes**  | Age                 | Integer     | 100             | ✓     | ✓     | ✓    |
|                        | Sex                 | Binary      | 2               | ✓     | ✓     | ✓    |
|                        | Diploma             | Categorical | 9               | ✓     | ✓     | ✓    |
|                        | Marital             | Categorical | 6               |       | ✓     | ✓    |
|                        | Cohabitation        | Binary      | 2               |       |       | ✓    |
|                        | Employment          | Categorical | 14              |       |       | ✓    |
|                        | Socioprofessional   | Categorical | 8               |       | ✓     | ✓    |
|                        | Activity            | Categorical | 18              |       | ✓     | ✓    |
|                        | Hours               | Categorical | 3               |       |       | ✓    |
|                        | Transport           | Categorical | 7               |       |       | ✓    |
|                        | ReferenceLink       | Categorical | 10              |       |       | ✓    |
|                        | FamilyLink          | Categorical | 5               |       | ✓     | ✓    |
| **Household attributes** | HouseholdSize      | Integer     | 18              | ✓     | ✓     | ✓    |
|                        | nChildren           | Integer     | 13              |       |       | ✓    |
|                        | nRooms              | Integer     | 20              |       |       | ✓    |
|                        | Surface             | Integer     | 7               |       |       | ✓    |
|                        | Parking             | Binary      | 2               |       |       | ✓    |
|                        | nCars               | Integer     | 4               |       | ✓     | ✓    |
|                        | Accommodation       | Categorical | 7               |       | ✓     | ✓    |
|                        | Household           | Categorical | 5               |       | ✓     | ✓    |
|                        | Occupancy           | Categorical | 6               |       |       | ✓    |
| **Geographical**       | Department          | Categorical | 8               | ✓     |       |      |
|                        | County              | Categorical | 181             |       | ✓     |      |
|                        | City                | Categorical | 416             |       |       |      |
|                        | TRIRIS              | Categorical | 1350            |       |       | ✓    |
|                        | IRIS                | Categorical | 4315            |       |       | ✓    |


# Note
- A data paper is in preparation for further details on the data collection, processing, and how to use it.
- Before launching the experiments, this folder should contain the unzip folders downloaded [here](https://data.mendeley.com/datasets/p2gcy7x7sd/)``datasets_Individual_0_03`` for the 0.03% case ``datasets_Individual_1`` for the 1% case.


# Example of results on the given dataset

Using the socio attribute set with 1% of the total population, the following results were obtained. 

[Experiments conducted June 17th, 2026]

| Model                   | Parameters  | $\overline{SRMSE}_3$           | $SSCIOD$ | Wasserstein-DCR |
|-----------------------------|------------------------------|--------|-----------------|
| Diffusion                   | Dimension: 500               | 1.17   | 7.33%           | 8.51e-07 |
| beta VAE                    | Dimension: 256 <br>Beta: 1   | 1.08   | 12.44%          | 2.67e-06 |
| beta VAE (embedding)        | Dimension: 256 <br>Beta: 0.1 | 2.24   | 12.38%          | 8.51e-07 |
| WGAN                        | Dimension: 201               | 1.37   | 6.74%           | 9.59e-07 |
| WGAN (embedding)            | Dimension: 200               | 1.5    | 5.89%           | 7.75e-07 |
| Bayesian Network Hill Climb |                              | 0.526  | 0.60%           | 6.48e-07 |
| Bayesian Network Tree       |                              | 0.762  | 1.66%           | 6.97e-07 |