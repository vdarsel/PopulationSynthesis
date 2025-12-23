# Data Presentation

The code is with a publicly and full-scale dataset, that can be download here: [Data]{https://doi.org/10.17632/p2gcy7x7sd.1}. The code is designed to generate a population at the **individual-level**. 

The code is designed to handle data at the **individual-level** in the format of [Data]{https://doi.org/10.17632/p2gcy7x7sd.1}., and put in the Data folder.


## Training data size

The size of the available data varies depending on the data providers. 
In the published data, two data scenarios are presented that differ by the size of the training dataset:

- At 0.03% of the total population, this scenario mimics cases where the amount of data is minimal (Household Travel Survey at the national level) 0
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