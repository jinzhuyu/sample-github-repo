# Data Documentation

## Overview

This folder contains all datasets used in the project. The data are organized into two categories:

- **raw/**: Original, unmodified data
- **processed/**: Cleaned and feature-engineered data used for modeling

This separation ensures transparency and reproducibility of the data pipeline.

------------------------------------------------------------------------

## Folder Structure

    data/
    ├── raw/
    │   └── stormwater_events_sample.csv
    ├── processed/
    │   └── stormwater_events_features.csv

------------------------------------------------------------------------

## Raw Data

### File: `raw/stormwater_events_sample.csv`

**Description**
This is a synthetic dataset representing stormwater system observations during rainfall events. It is designed for instructional purposes and
mimics real-world hydrologic monitoring data.

**Variables**

  ------------------------------------------------------------------------

  Column Name                            Description                                            Units

  ---------------------------- ---------------------------- --------------

 rainfall_intensity                 Rainfall intensity during event               mm/hr                       

 antecedent_moisture        Soil moisture index prior to  event       unitless
 drainage_area                    Contributing drainage area                    hectares

slope                                    Average slope of drainage area             fraction
peak_flow                           Observed peak outflow                           m³/s  

  ------------------------------------------------------------------------

------------------------------------------------------------------------

## Processed Data

### File: `processed/stormwater_events_features.csv`

**Description**
This dataset is derived from the raw data through preprocessing and feature engineering.

------------------------------------------------------------------------

## Data Generation

    python src/make_dataset.py

------------------------------------------------------------------------
