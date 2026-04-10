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
![](/images/map_ISMN_stations.png)

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

## 🛠️ Tech Stack
- Python (3.12)
- Data manipulation, analysis, & visualization
  - pandas
  - NumPy
  - Matplotlib
  - Plotly
- Machine learning
  - tbd
- Jupyter Notebook 

## ▶️ How to Run

### Setup

You must have Python 3.12 installed. 


#### Conda

If using conda, create environment from `environment.yml` then install `src` as 
a package:

```
conda env create -f environment.yml
conda activate ft-soil
```

#### venv and pip

Otherwise, use virtual environment and `requirements.txt`:

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

## ℹ️ Sources

The raw ASCAT and ERA5 data files can be found at https://doi.org/10.5281/zenodo.19259521.

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

## 🙏 Acknowledgements

- Main Supervisor: Univ.Prof. Wolfgang Wagner
- Co-supervisor: Sebastian Hahn
- Co-supervisor: Prof. Nysret Musliu

## ©️ Licensing

The source code in this repository is licensed under the MIT License.

* ASCAT data are © [EUMETSAT OSI SAF](https://osi-saf.eumetsat.int/licensing-and-attribution) and licensed under CC-BY-4.0.
* ERA5 data are © [Copernicus Climate Change Service / ECMWF](https://www.ecmwf.int/en/forecasts/accessing-forecasts/service-agreements) and licensed under CC-BY-4.0.

The data are not covered by the MIT license.