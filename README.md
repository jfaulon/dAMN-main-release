# dAMN — Dynamic Artificial Metabolic Network  
Hybrid Neural Network Framework for Dynamic Flux Balance Analysis (dFBA)

This repository contains the reference implementation of dAMN, a hybrid modelling framework that combines:
- a neural network predicting effective intracellular flux distributions from initial media composition, and  
- a dynamic FBA (dFBA) engine simulating extracellular metabolite and biomass trajectories.

dAMN supports both OD-based and concentration-based time-series datasets, automatically detected by the loader.

---

## 1. Repository Structure

The project is organised as follows:

    data/                     Experimental datasets (media tables, OD files, concentration files, SBML model)
    figure/                   Training curves, prediction plots, diagnostics (auto-generated)
    model/                    Trained model weights, validation arrays, run outputs
    model_xfold/              Outputs from cross-validation or repeated training runs
    __pycache__/              Python cache directory

    dAMN.ipynb                Main notebook: data loading, training, prediction, visualisation
    dAMN_parameter_search.py  Parameter and hyperparameter search utility
    data.py                   Data loading, automatic format detection, preprocessing, structure building
    model.py                  Neural network definition and dFBA integration routines
    plot.py                   Plotting utilities
    utils.py                  Auxiliary functions (loss terms, normalization, batching, etc.)
    environment.yaml          Conda environment specification
    LICENSE                   License information
    README.md                 Project documentation
    README.rtf                Legacy documentation (superseded by this file)

---

## 2. Supported Data Formats

The dAMN framework supports two mutually exclusive types of experimental time-series datasets.
Both formats rely on a common media file and must be placed in the `data/` directory.

The two supported dataset types are:

1. OD-based time series (e.g., M28 dataset)  
2. Concentration-based time series (e.g., Millard dataset)

The system automatically detects which format is being used.

---

### 2.1 Media File Format (Common to All Datasets)

Initial extracellular conditions for each experiment are specified in a CSV file with the following structure:

    ID, <met1>, <met2>, ..., <metN>

Rules:
- `ID` uniquely identifies an experiment.
- All other columns correspond to extracellular metabolites from the COBRA model.
- `BIOMASS` must not appear in this table; it is appended internally as the last species.
- One row per experiment.

This media file is used to initialise extracellular metabolite concentrations before each dFBA simulation.

---

## 3. OD-Based Time-Series Format

This format is used when only optical density (OD) measurements are available.  
dAMN converts OD values into biomass concentration using internal transformation functions.

A typical file for experiment X contains:

    T_X, OD_X, DEV_X

Where:
- `T_X`  = time points  
- `OD_X` = OD measurements (raw or log-transformed)  
- `DEV_X` = optional measurement deviation  

Characteristics:
- Only BIOMASS is experimentally observed.
- OD is exponentiated when required and mapped to biomass internally.
- The loader automatically detects this format when any column contains `OD_` or `DEV_`.

Internal representation:
- If the media file defines `k` metabolites, the internal dynamic vector becomes  
  `[media_metabolites..., BIOMASS]`
- BIOMASS is always stored at the last index.

---

## 4. Concentration-Based Time-Series Format

This format is used when extracellular metabolite concentrations are directly measured.

Required CSV structure:

    time, <met1>, <met2>, ..., BIOMASS

Rules:
- `BIOMASS` must be present and must be the last column.
- Column names must match extracellular metabolite IDs in the COBRA model.
- Units must be consistent with the media file.
- Any number of metabolites may be measured.

Internal structure:
- If metabolites are `[GLC, ACE, ...]`, the internal state vector becomes  
  `[GLC, ACE, ..., BIOMASS]`.

The total dimension equals the number of measured metabolites.

---

## 5. Automatic Format Detection

The dataset type is determined automatically by the loader (`process_data()` in `data.py`).

Detection rules:
- If any column name contains `OD_` or `DEV_` → OD-based dataset.
- Otherwise, if valid metabolite columns exist and the last column is `BIOMASS` → concentration-based dataset.

This mechanism ensures seamless switching between dataset formats.

---

## 6. Critical Internal Convention

Regardless of dataset format:

    BIOMASS MUST BE THE LAST METABOLITE.

This requirement is essential because:
- Multiple loss terms operate specifically on the last index.
- The biomass reaction is identified based on this position.
- Monotonicity checks and state-transition constraints rely on BIOMASS being last.

Violating this ordering will break training and simulation.

---

## 7. Using dAMN

### 7.1 Environment Setup

    conda env create -f environment.yaml
    conda activate dAMN_env

### 7.2 Running the Model

Place all dataset files in the `data/` directory.

Open the notebook `dAMN.ipynb` to:

- load and validate datasets  
- train the neural network + dFBA model  
- run dynamic predictions  
- visualise trajectories and diagnostics  
- export figures to the `figure/` directory  

### 7.3 Parameter Search

To run a parameter or hyperparameter search:

    python dAMN_parameter_search.py

Outputs are saved in:

- `model/` for single-run results  
- `model_xfold/` for cross-validation or repeated trainings  
- `figure/` for all generated plots  

---

## 8. Comparison of Dataset Types

| Feature                      | OD-Based Format         | Concentration-Based Format      |
|-----------------------------|--------------------------|---------------------------------|
| Input values                | OD measurements          | Metabolite concentrations       |
| Observed metabolites        | BIOMASS only             | One or more metabolites         |
| BIOMASS column              | Computed internally      | Required, must be last column   |
| Multiple experiments        | Via experiment ID        | One file or structured tables   |
| OD → biomass mapping        | Required                 | Not applicable                  |

---

## 9. Contributors

- Jean-Loup Faulon (INRAE) — Conceptualization, methodology, core model design  
- Danilo Dursoniah (INRAE) — Implementation, dataset integration, training pipeline, maintenance

---

## 10. License

This project is distributed under the terms of the `LICENSE` file.
