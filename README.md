# Prediction of Freeze/Thaw Transitions in Soil

## 📌 TL;DR
Comparison of machine learning (ML) driven analysis of Advanced Scatterometer (ASCAT) backscatter vs ERA5 data in 
predicting freeze/thaw (F/T) transitions in soil, validated against international soil moisture network (ISMN) ground 
observations.

## 🚀 Overview

This project investigates whether ML models applied to ASCAT backscatter time-series can accurately predict soil F/T 
transitions, and how these predictions compare to those derived from ERA5 reanalysis data.

Freeze/thaw transitions are identified using in-situ soil temperature observations from the ISMN, which serve as 
reference ground truth. For a set of selected stations, ASCAT backscatter observations are processed into time-series 
features and used to train ML models that predict daily frozen or thawed soil states. Transition dates are then 
extracted and compared against:

- Ground-based ISMN observations (reference)

- ERA5-derived soil temperature transitions (baseline)

The primary research question guiding this project is:

> How does ML-processed ASCAT data compare to ERA5 in accurately predicting soil freeze/thaw transitions?

A set of 10 ISMN stations was created, and the locations of the stations are depicted below.

![](/images/for_README/map_ISMN_stations.png)

**Repository Structure:**

```angular2html
├── ISMN_site_survey.csv
├── requirements.txt
├── notebooks
│   ├── rolling_classification
│   │   ├── 1_ISMN_data_cleaning.ipynb
│   │   ├── 2_ASCAT_ERA5_data_cleaning.ipynb
│   │   ├── 3_assess_methodology.ipynb
│   │   ├── 4_model_selection.ipynb
│   │   └── 5_model_vs_ERA5.ipynb
│   └── simple_classification
│       ├── 1_ISMN_data_cleaning.ipynb
│       ├── 2_ASCAT_ERA5_data_cleaning.ipynb
│       ├── 3_assess_methodology.ipynb
│       ├── 4_model_selection.ipynb
│       └── 5_model_vs_ERA5.ipynb
└── src
    └── freeze_thaw
        ├── data_preparation
        │   ├── ascat_era5.py
        │   ├── feature_engineering.py
        │   ├── general.py
        │   ├── ismn.py
        │   ├── nan_handling.py
        │   ├── outlier_detection.py
        │   └── splitting.py
        ├── data_understanding
        │   └── visualization.py
        ├── evaluation
        │   ├── metrics.py
        │   └── statistics.py
        ├── modeling
        │   ├── lgb_test.py
        │   └── lgb_train.py
        ├── _internal_functions.py
        ├── config.py
        ├── utils.py
        └── validation.py
```

## 🎯 Motivation

Soil F/T dynamics play a crucial role in:

- Infrastructure stability (frost heave, subsidence)

- Agricultural productivity and planting cycles

- Hydrological processes and runoff generation

- Permafrost monitoring and climate change assessment

Ground-based measurements can determine F/T transitions with high accuracy, but their spatial coverage is sparse and 
uneven globally. In contrast, satellite remote sensing offers consistent global observations. The ASCAT microwave 
scatterometer measures surface backscatter, which is sensitive to changes in soil dielectric properties associated with 
freezing and thawing. ML may enable direct extraction of F/T transitions from these backscatter time series.

ERA5 is a widely-used source that provides model-based soil temperature estimates derived from data assimilation. 
Possible downsides of these reanalysis products are that they may smooth or misrepresent local transition timing.

This project evaluates whether ML applied to ASCAT backscatter can match or outperform ERA5 in predicting soil 
freeze/thaw transitions, providing a reproducible and observation-driven alternative for large-scale monitoring.

## 📊 Results

### Simple Classification

The first set of results are based on classifying an observation solely on its ISMN soil temperature.
Where the boundaries are defined as `-1 °C < transition ≤ 1 °C`, with frozen and thawed being below and 
above these bounds respectively. For more details, see `notebooks/simple_classification/5_model_vs_ERA5.ipynb`.
The differences shown are the lightGBM minus ERA5 F1 scores.

#### Macro F1

| Station          | lightGBM-ASCAT | ERA5  | Difference |
|------------------|----------------|-------|------------|
| Aberdeen-35-WNW  | 0.641          | 0.650 | -0.009     |
| Jamestown-38-WSW | 0.746          | 0.608 | 0.138      |
| GobblersKnob     | 0.830          | 0.746 | 0.084      |
| Nenana           | 0.789          | 0.651 | 0.138      |
| L23              | 0.871          | 0.568 | 0.303      |
| L38              | 0.794          | 0.523 | 0.271      |
| NST-07           | 0.740          | 0.620 | 0.120      |
| NST-09           | 0.824          | 0.579 | 0.245      |
| SOD012           | 0.667          | 0.549 | 0.118      |
| SOD103           | 0.556          | 0.326 | 0.230      |

#### Transition F1

| Station          | lightGBM-ASCAT | ERA5  | Difference |
|------------------|----------------|-------|------------|
| Aberdeen-35-WNW  | 0.576          | 0.340 | 0.235      |
| Jamestown-38-WSW | 0.642          | 0.278 | 0.363      |
| GobblersKnob     | 0.640          | 0.462 | 0.178      |
| Nenana           | 0.641          | 0.281 | 0.360      |
| L23              | 0.716          | 0.091 | 0.625      |
| L38              | 0.561          | 0.050 | 0.511      |
| NST-07           | 0.530          | 0.311 | 0.219      |
| NST-09           | 0.652          | 0.169 | 0.483      |
| SOD012           | 0.821          | 0.406 | 0.415      |
| SOD103           | 0.779          | 0.146 | 0.633      |

### Rolling Classification

The second set of results are based on classifying an observation based on its current ISMN soil temperature along with
the temperatures for the preceding 72 hours. An observation is classified as frozen if all temperatures are at or below 
0 °C. Vice versa for the thawed state. The transition state is if the temperatures cross the boundary. For more details, 
see `notebooks/rolling_classification/5_model_vs_ERA5.ipynb`. The differences shown are the lightGBM minus ERA5 F1 scores.

#### Macro F1

| Station          | lightGBM-ASCAT | ERA5  | Difference |
|------------------|----------------|-------|------------|
| Aberdeen-35-WNW  | 0.651          | 0.602 | 0.048      |
| Jamestown-38-WSW | 0.615          | 0.612 | 0.004      |
| GobblersKnob     | 0.714          | 0.660 | 0.054      |
| Nenana           | 0.693          | 0.580 | 0.113      |
| L23              | 0.654          | 0.502 | 0.152      |
| L38              | 0.791          | 0.484 | 0.307      |
| NST-07           | 0.737          | 0.656 | 0.081      |
| NST-09           | 0.765          | 0.567 | 0.198      |
| SOD012           | 0.525          | 0.502 | 0.023      |
| SOD103           | 0.348          | 0.244 | 0.104      |

#### Transition F1

| Station          | lightGBM-ASCAT | ERA5  | Difference |
|------------------|----------------|-------|------------|
| Aberdeen-35-WNW  | 0.196          | 0.170 | 0.026      |
| Jamestown-38-WSW | 0.011          | 0.146 | -0.135     |
| GobblersKnob     | 0.248          | 0.197 | 0.051      |
| Nenana           | 0.172          | 0.008 | 0.165      |
| L23              | 0.477          | 0.139 | 0.339      |
| L38              | 0.660          | 0.084 | 0.576      |
| NST-07           | 0.503          | 0.340 | 0.163      |
| NST-09           | 0.593          | 0.224 | 0.369      |
| SOD012           | 0.087          | 0.101 | -0.015     |
| SOD103           | 0.120          | 0.000 | 0.120      |

The results suggest that the lightGBM model trained on the ASCAT data significantly outperforms ERA5. However, there's
a weakness in the methodology that may be unfairly punishing ERA5. Specifically, large area measurements are being 
evaluated against a single point measurment from an ISMN station. The lightGBM model trained on ASCAT is also affected
by this, but the model is specifically trained against labels derived from a single ISMN point measurement. A better 
approach may be to select multiple ISMN stations that fall under one ASCAT and ERA5 footprint, then taking the average
temperature. See `notebooks/../3_assess_methodology.ipynb` for more details.

## 🛠️ Tech Stack
- Python (3.12)
- Pydantic
- Data manipulation, analysis, & visualization
  - pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Plotly
- Model training & evaluation
  - LightGBM
  - Scikit-learn
- Jupyter Notebook 

## ▶️ How to Run

### Setup

It's recommended to use Python 3.12. Create virtual environment: 

```
python3.12 -m venv .venv
```

If using Mac/Linux:

```
source .venv/bin/activate
```

If using Windows:

```
.venv\Scripts\activate
```

Install requirements:

```
pip install -r requirements.txt
```

- Download the raw ASCAT and ERA5 data files from 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19259520.svg)](https://doi.org/10.5281/zenodo.19259520)
and move them to `../data/raw/ASCAT_ERA5`. 
- Download the ISMN data and move them to `../data/raw/ISMN/{station_name}`.
- Run notebooks 1-4. 
- Execute `../src/freeze_thaw/modeling/lgb_train.py`; be mindful of the parameters under `if __name__ == '__main__':`
- Run notebook 5.

## ℹ️ Sources

The raw ASCAT and ERA5 data files can be found at 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19259520.svg)](https://doi.org/10.5281/zenodo.19259520).

The individual swath orbit files from the **ASCAT** Surface Soil Moisture
(SSM) Climate Data Record (CDR) v8 at 12.5 km sampling (H121,
https://doi.org/10.15770/EUM_SAF_H_0011 ) were stacked and converted
into time series format. Since the data are provided on a fixed Earth
grid, this processing step involved only restructuring the data without
altering the original values. For each in situ station, the time series
of the nearest ASCAT grid point was extracted. During this
transformation, data from the corresponding Intermediate Climate Data
Record (ICDR, H139) were appended as well.

**ERA5** data has been downloaded from the Copernicus Climate Data Store
(CDS) (https://doi.org/10.24381/cds.adbb2d47) and converted into time
series format. The ERA5 data are provided on a 0.25 degree grid and data
has not been altered during data conversion. The time series of the
closest grid point of the ERA5 dataset has been extracted of each in
situ station.

[ISMN T&Cs](https://ismn.earth/en/terms-and-conditions/) forbids the re-export or transfer 
of the original data to third parties. Therefore, the ISMN data used for this analysis is 
not included in this repo or elsewhere. The data used for the analysis can be downloaded 
from the data provider using the following steps:
1. Create an account at [ismn.earth](https://ismn.earth/en/).
2. Click on "Data Access" on the home page, then set the initial filters:

![](/images/for_README/ISMN_initial_filters.png)

3. Search for stations:

![](/images/for_README/station_example.png)

4. Create and execute an area filter around the station of interest, then click "Download" using
the four steps depicted below:

![](/images/for_README/area_filter.png)

5. Select the following parameters, then click on "your requests" to download the data.

![](/images/for_README/download.png)

## 🙏 Acknowledgements

- Main Supervisor: Univ.Prof. Wolfgang Wagner
- Co-supervisor: Sebastian Hahn
- Co-supervisor: Prof. Nysret Musliu

## ©️ Licensing

The source code in this repository is licensed under the MIT License.

* ASCAT data are © [EUMETSAT OSI SAF](https://osi-saf.eumetsat.int/licensing-and-attribution) and licensed under CC-BY-4.0.
* ERA5 data are © [Copernicus Climate Change Service / ECMWF](https://www.ecmwf.int/en/forecasts/accessing-forecasts/service-agreements) and licensed under CC-BY-4.0.

The data are not covered by the MIT license.