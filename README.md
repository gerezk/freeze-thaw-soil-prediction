# Prediction of Freeze/Thaw Transitions in Soil

## 📌 TL;DR
Comparison of machine learning (ML) driven analysis of Advanced Scatterometer (ASCAT) backscatter vs ERA5 data in predicting freeze/thaw (F/T) transitions in soil, validated against international soil moisture network (ISMN) ground observations.

## 🚀 Overview

This project investigates whether ML models applied to ASCAT backscatter time-series can accurately predict soil F/T transitions, and how these predictions compare to those derived from ERA5 reanalysis data.

Freeze/thaw transitions are identified using in-situ soil temperature observations from the ISMN, which serve as reference ground truth. For a set of selected stations, ASCAT backscatter observations are processed into time-series features and used to train ML models that predict daily frozen or thawed soil states. Transition dates are then extracted and compared against:

- Ground-based ISMN observations (reference)

- ERA5-derived soil temperature transitions (baseline)

The primary research question guiding this project is:

> How does ML-processed ASCAT data compare to ERA5 in accurately predicting soil freeze/thaw transitions?

The ultimate goal is to evaluate whether physically grounded, observation-driven satellite backscatter combined with ML can match or outperform a global reanalysis product in detecting critical soil thermal state transitions.

## 🎯 Motivation

Soil F/T dynamics play a crucial role in:

- Infrastructure stability (frost heave, subsidence)

- Agricultural productivity and planting cycles

- Hydrological processes and runoff generation

- Permafrost monitoring and climate change assessment

Ground-based measurements can determine F/T transitions with high accuracy, but their spatial coverage is sparse and uneven globally. In contrast, satellite remote sensing offers consistent global observations. The ASCAT microwave scatterometer measures surface backscatter, which is sensitive to changes in soil dielectric properties associated with freezing and thawing. ML may enable direct extraction of F/T transitions from these backscatter time series.

ERA5 is a widely-used source that provides model-based soil temperature estimates derived from data assimilation. Possible downsides of these reanalysis products are that they may smooth or misrepresent local transition timing.

This project evaluates whether ML applied to ASCAT backscatter can match or outperform ERA5 in predicting soil freeze/thaw transitions, providing a reproducible and observation-driven alternative for large-scale monitoring.

## 🛠️ Tech Stack
- Python (3.12)
- Data manipulation, analysis, & visualization
  - pandas
  - NumPy
  - Matplotlib
- Machine learning
  - tbd
- Jupyter Notebook 

## ℹ️ Sources

## ©️ Licensing

The source code in this repository is licensed under the MIT License.

* ASCAT data are © [EUMETSAT OSI SAF](https://osi-saf.eumetsat.int/licensing-and-attribution) and licensed under CC-BY-4.0.
* ERA5 data are © [Copernicus Climate Change Service / ECMWF](https://www.ecmwf.int/en/forecasts/accessing-forecasts/service-agreements) and licensed under CC-BY-4.0.

The data are not covered by the MIT license.