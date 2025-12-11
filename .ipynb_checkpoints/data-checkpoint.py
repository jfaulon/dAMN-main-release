from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf
import pandas as pd
import cobra
import json
import matplotlib.pyplot as plt
import sklearn
import utils

###############################################################################
# PROCESS DATA: MEDIA, OD, AND COBRA MODEL
###############################################################################

def parse_training_mode(mode_str, default_split="forecast", verbose=True):
    """Normalize a high-level training *mode* string into the internal
    `train_test_split` value.

    Parameters
    ----------
    mode_str : str
        User-facing mode string. Typical examples:
            - 'medium'
            - 'forecast'
            - 'concentration-GLC'
            - 'concentration-ACE'
            - 'concentration-BIOMASS'
            - 'concentration-' + full metabolite id
    default_split : {'forecast', 'medium'}, optional
        Fallback value if the mode string is not recognized. Defaults to 'forecast'.
    verbose : bool, optional
        If True, prints how the mode was interpreted.

    Returns
    -------
    train_test_split : str
        The value that should be passed around internally.
    """
    mode_str = str(mode_str).strip()

    # Simple cases
    if mode_str in ("forecast", "medium"):
        if verbose:
            print(f"[parse_training_mode] mode='{mode_str}' -> train_test_split='{mode_str}'")
        return mode_str

    # Concentration hold-out modes stay as-is; MetabolicModel will interpret the
    # 'concentration-*' pattern.
    if mode_str.lower().startswith("concentration-"):
        if verbose:
            print(f"[parse_training_mode] mode='{mode_str}' kept as concentration mode.")
        return mode_str

    # Fallback
    if verbose:
        print(
            f"[parse_training_mode] Warning: unrecognized mode '{mode_str}'. "
            f"Falling back to default_split='{default_split}'."
        )
    return default_split

def get_holdout_index_from_split(train_test_split: str, metabolite_ids: List[str]):
    """Return index of the metabolite held out from concentration loss.
    This interprets train_test_split strings of the form 'concentration-XXX'.
    For example:
        'concentration-GLC'      -> metabolite 'glc__D_e' (if present)
        'concentration-ACE'      -> metabolite 'ac_e'
        'concentration-BIOMASS'  -> metabolite 'BIOMASS'

    If train_test_split does not encode a concentration hold-out, returns None.
    """
    if train_test_split is None:
        return None

    tts = str(train_test_split).strip()
    if not tts.lower().startswith("concentration-"):
        return None

    # Suffix after 'concentration-'
    suffix = tts.split("-", 1)[1].strip()

    # Map short names to canonical metabolite IDs used in the COBRA model
    short_to_full = {
        "GLC": "glc__D_e",
        "ACE": "ac_e",
        "BIOMASS": "BIOMASS",
    }
    metabolite_id = short_to_full.get(suffix.upper(), suffix)

    try:
        return metabolite_ids.index(metabolite_id)
    except ValueError:
        raise ValueError(
            f"Concentration split requested for '{metabolite_id}', "
            f"but this metabolite is not in metabolite_ids: {metabolite_ids}"
        )
        
def build_Stoichiometry_and_Transport_from_cobra(
    cobra_model_file: str, 
    medium_met_ids: List[str], 
    biomass_rxn_id: str,
    verbose=False
) -> Tuple[np.ndarray, np.ndarray, int, int, List[str]]:
    """
    Build stoichiometric and Transport matrices from a COBRA model.

    medium_met_ids:
        List of external/metabolite IDs you want as "state variables" in the
        model (e.g. ['glc__D_e', 'ac_e', 'BIOMASS']).
    """
    model = cobra.io.read_sbml_model(cobra_model_file)
    
    # Get reaction and metabolite IDs (strings)
    rxn_ids = [rxn.id for rxn in model.reactions]
    full_met_ids = [met.id for met in model.metabolites]

    m = len(full_met_ids)
    n = len(rxn_ids)

    # Build stoichiometric matrix
    Stoichiometry = np.zeros((m, n), dtype=np.float32)
    for j, rxn in enumerate(model.reactions):
        for met, coeff in rxn.metabolites.items():
            i = full_met_ids.index(met.id)
            Stoichiometry[i, j] = coeff
            
    # Build Transport matrix (rows = medium_met_ids, cols = reactions)
    Transport = np.zeros((len(medium_met_ids), n), dtype=np.float32)
    for col_idx, rxn_obj in enumerate(model.reactions):
        rxn_id = rxn_obj.id

        # Biomass "reaction" row
        if rxn_id == biomass_rxn_id:
            if 'BIOMASS' in medium_met_ids:
                row_idx = medium_met_ids.index('BIOMASS')
                Transport[row_idx, col_idx] = 1.0
                if verbose:
                    print(f'Transport[{row_idx},{col_idx}] = 1.0  BIOMASS')
            continue

        # Exchange reactions, e.g. EX_glc__D_e(o/i)
        if rxn_id.startswith('EX_'):
            io = rxn_id[-1:]      # 'o' or 'i'
            met_id_candidate = rxn_id[3:-2]  # EX_glc__D_e -> glc__D
            if met_id_candidate in medium_met_ids:
                row_idx = medium_met_ids.index(met_id_candidate)
                Transport[row_idx, col_idx] = -1.0 if io == 'i' else 0 # !!! 
                if met_id_candidate == 'ac_e':
                    Transport[row_idx, col_idx] = -1.0 if io == 'i' else 1
                if verbose and Transport[row_idx, col_idx]:
                    print(f'Transport[{row_idx},{col_idx}] = {Transport[row_idx, col_idx]}  {rxn_id} {met_id_candidate}')

    if verbose:
        print(f'Transport shape: {Transport.shape}, Stoichiometry shape: {Stoichiometry.shape}')
        print(f'Number of metabolites (k): {len(medium_met_ids)}, Number of fluxes (n): {n}')
        
    return Stoichiometry, Transport, m, n, rxn_ids

def process_data(
    media_file: str,
    od_file: str,
    cobra_model_file: str,
    biomass_rxn_id: str,
    verbose=False
) -> Tuple[np.ndarray, List[str], Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, np.ndarray, List[str]]:
    """
    Process media, OD/CONC, and COBRA model to generate structured experimental data and matrices.

    Handles two cases:
      - OD mode (M28_*): od_file has columns like T_1, OD_1, DEV_1, T_2, ...
      - CONC mode (Millard_*): od_file has T_1, GLC_1, DEV_1, ACE_1, DEV_1.1, BIOMASS_1, DEV_1.2, ...
    """
    media_df = pd.read_csv(media_file)
    od_df = pd.read_csv(od_file)

    # Detect OD vs concentration mode from column names
    cols_upper = [c.upper() for c in od_df.columns]
    is_od_mode = any("OD" in c for c in cols_upper)

    # ---- Common: define metabolite_ids from media + BIOMASS ----
    # Use all media metabolites (skip the first column 'ID') and add BIOMASS at the end.
    metabolite_ids = list(media_df.columns[1:])
    if 'BIOMASS' not in metabolite_ids:
        metabolite_ids.append('BIOMASS')

    # ---- Define time grid ----
    # For OD: union of all T_* columns (what you had before).
    # For CONC: we can safely use the first T_* column.
    t_cols = [c for c in od_df.columns if c.startswith('T_')]
    if not t_cols:
        raise ValueError("No time columns like 'T_<id>' found in measurement file.")

    if is_od_mode:
        # Original behaviour: all unique times from T_* columns
        times_all = pd.unique(od_df[t_cols].values.flatten())
        times = np.array(sorted(t for t in times_all if not pd.isnull(t)), dtype=float)
    else:
        # Concentration mode: just use the time vector from the first T_* column
        times = od_df[t_cols[0]].to_numpy(dtype=float)

    # ---- Prepare containers ----
    experiment_data: Dict[int, np.ndarray] = {}
    dev_data: Dict[int, np.ndarray] = {}

    # Which experiment IDs are actually present in the measurement file?
    # We detect them from T_<id> columns.
    present_ids = set()
    for c in t_cols:
        try:
            _, eid_str = c.split('_', 1)
            present_ids.add(int(eid_str))
        except Exception:
            pass

    # A) OD MODE (M28_*)
    if is_od_mode:
        if verbose:
            print("[process_data] detected OD mode (M28-style OD + DEV)")

        for exp_id in media_df['ID']:
            if int(exp_id) not in present_ids:
                # No T_exp_id, OD_exp_id columns => skip this experiment
                continue

            t_col  = f'T_{int(exp_id)}'
            od_col = f'OD_{int(exp_id)}'
            dev_col = f'DEV_{int(exp_id)}'

            if t_col not in od_df.columns or od_col not in od_df.columns:
                raise ValueError(f'Missing columns for experiment ID {exp_id} in OD file.')

            od_times = od_df[t_col].values
            log_od_values = od_df[od_col].values
            log_od_dev = od_df[dev_col].values
            dev_data[int(exp_id)] = log_od_dev  # keep as in original code

            od_values = np.exp(log_od_values)
            biomass_concentration = utils.OD_to_concentration(od_values)

            # Matrix: (T, k) with NaNs, only biomass column filled
            conc_matrix = np.full((len(times), len(metabolite_ids)), np.nan, dtype=np.float32)

            # Set initial media concentrations (excluding biomass)
            row_in_media = media_df[media_df['ID'] == exp_id].iloc[0, 1:].values
            conc_matrix[0, :-1] = row_in_media

            # Set biomass over time by matching od_times to global 'times'
            # Biomass index is last column
            bio_idx = len(metabolite_ids) - 1

            # If 0 in od_times, use that as t0, else first measured
            if 0 in od_times:
                idx_0 = np.where(od_times == 0)[0][0]
                conc_matrix[0, bio_idx] = biomass_concentration[idx_0]
            else:
                conc_matrix[0, bio_idx] = biomass_concentration[0]

            for i_time, t_val in enumerate(times[1:], start=1):
                if t_val in od_times:
                    idx_t = np.where(od_times == t_val)[0][0]
                    conc_matrix[i_time, bio_idx] = biomass_concentration[idx_t]

            experiment_data[int(exp_id)] = conc_matrix

    # B) CONCENTRATION MODE (Millard_*): GLC, ACE, BIOMASS already in conc.
    else:
        if verbose:
            print("[process_data] detected concentration mode (Millard-style GLC/ACE/BIOMASS)")

        # Map model metabolite IDs → column prefixes in the Millard file
        # This is specific to your current files:
        #   glc__D_e -> GLC_<id>
        #   ac_e     -> ACE_<id>
        #   BIOMASS  -> BIOMASS_<id>
        def _col_for_met(exp_id: int, met_id: str) -> str:
            if met_id == 'BIOMASS':
                return f'BIOMASS_{exp_id}'
            elif met_id == 'glc__D_e':
                return f'GLC_{exp_id}'
            elif met_id == 'ac_e':
                return f'ACE_{exp_id}'
            else:
                # For now, we have only these three; others would be NaN.
                return None

        # For now, only experiments that actually appear in measurement file
        # (Millard_12_1 has only ID=1).
        for exp_id in media_df['ID']:
            if int(exp_id) not in present_ids:
                continue  # no T_<id> / GLC_<id> etc. for this experiment

            t_col = f'T_{int(exp_id)}'
            if t_col not in od_df.columns:
                raise ValueError(f"Missing time column {t_col} for experiment {exp_id}")

            # time vector for this experiment (should match 'times')
            local_times = od_df[t_col].to_numpy(dtype=float)
            if len(local_times) != len(times) or not np.allclose(local_times, times):
                # If they differ, you could resample/align; for now we enforce equality.
                raise ValueError(f"Time grid mismatch for experiment {exp_id}")

            # Allocate (T, k) and fill from measured GLC/ACE/BIOMASS
            T = len(times)
            k = len(metabolite_ids)
            conc_matrix = np.full((T, k), np.nan, dtype=np.float32)

            for j, met_id in enumerate(metabolite_ids):
                col_name = _col_for_met(int(exp_id), met_id)
                if col_name is not None and col_name in od_df.columns:
                    conc_matrix[:, j] = od_df[col_name].to_numpy(dtype=np.float32)
                else:
                    # Leave as NaN (unmeasured species)
                    pass

            # Store biomass dev only (1D) if we find a DEV column for BIOMASS
            dev_col_candidates = [c for c in od_df.columns
                                  if c.startswith(f'DEV_{int(exp_id)}') and 'BIOMASS' not in c.upper()]
            # In Millard_12_1 we have DEV_1, DEV_1.1, DEV_1.2; biomass dev is DEV_1.2
            biomass_dev_col = None
            for c in od_df.columns:
                if c.startswith(f'DEV_{int(exp_id)}') and c.endswith('.2'):
                    biomass_dev_col = c
                    break
            if biomass_dev_col is not None:
                dev_data[int(exp_id)] = od_df[biomass_dev_col].to_numpy(dtype=np.float32)
            else:
                dev_data[int(exp_id)] = np.zeros(T, dtype=np.float32)

            experiment_data[int(exp_id)] = conc_matrix

    # ---- Build stoichiometric and transport matrices from COBRA ----
    Stoichiometry, Transport, _, _, rxn_ids = build_Stoichiometry_and_Transport_from_cobra(
        cobra_model_file, metabolite_ids, biomass_rxn_id, verbose=verbose
    )

    return np.array(times, dtype=np.float32), metabolite_ids, experiment_data, dev_data, Stoichiometry, Transport, rxn_ids


def prepare_experiment_array(
    T: int,
    metabolite_ids: List[str],
    experiment_data: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Flatten experimental data into a 2D array with shape (Z, T*k).
    This exactly reproduces the old behavior:
    - Always take the FIRST T rows
    - Always take ALL columns (all metabolites)
    - Always do row-major flattening
    """
    k = len(metabolite_ids)
    all_flat = []

    for exp_id in experiment_data.keys():
        conc_matrix = experiment_data[exp_id]

        # Old behavior: ALWAYS take first T rows
        row_flat = conc_matrix[:T, :].reshape(-1)

        all_flat.append(row_flat)

    return np.stack(all_flat, axis=0)
    
def prepare_experiment_array(
    time_ids: np.ndarray,
    metabolite_ids: List[str],
    experiment_data: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Flatten experimental data into a 2D array with shape:
        (num_experiments, T * k)

    time_ids:
        - either an integer T (number of time steps),
        - or a 1D array of times of length T.
    """
    k = len(metabolite_ids)
    # Normalize time_ids → T (number of rows to keep)
    if isinstance(time_ids, (int, np.integer)):
        T = int(time_ids)
    else:
        T = len(time_ids)
    all_flat = []
    for exp_id in experiment_data.keys():
        conc_matrix = experiment_data[exp_id]          # shape (T_full, k)
        # Keep first T rows (you use the same time grid for all experiments)
        conc_slice = conc_matrix[:T, :]                # shape (T, k)
        row_flat = conc_slice.reshape(-1)              # (T*k,)
        all_flat.append(row_flat)

    return np.stack(all_flat, axis=0)

def build_dfba_inputs(
    media_file,
    od_file,
    cobra_model_file,
    biomass_rxn_id,
    verbose=False,
):
    """
    Build full time-series and concentration arrays for ALL experiments,
    without any train/validation split and without creating a MetabolicModel.

    This is the natural entry point for dFBA-style simulations.

    Returns
    -------
    times : np.ndarray, shape (T,)
        Time points (same for all experiments).
    metabolite_ids : list of str, length k
        External metabolites + BIOMASS (same ordering as in the arrays).
    experiment_array : np.ndarray, shape (Z, T, k)
        Full concentration trajectories for each experiment (Z), all time
        points (T) and all metabolites (k), constructed as in
        `prepare_experiment_array`.
    dev_array : np.ndarray or None
        Dev data stacked as an array if available, otherwise None.
    exp_ids : list
        List of experiment IDs (keys of experiment_data) in the same order
        as the first dimension of `experiment_array`.
    Stoichiometry : np.ndarray
        Stoichiometric matrix returned by `process_data`.
    Transport : np.ndarray
        Transport matrix returned by `process_data`.
    rxn_ids : list of str
        Reaction IDs returned by `process_data`.
    """

    # 1) Parse input files with the same logic as in create_model_train_val
    times, metabolite_ids, experiment_data, dev_data, \
        Stoichiometry, Transport, rxn_ids = process_data(
            media_file,
            od_file,
            cobra_model_file,
            biomass_rxn_id,
            verbose=verbose,
        )

    # 2) Fix experiment ordering and build the full (Z, T, k) array
    exp_ids = list(experiment_data.keys())
    # keep order as-is, or sort if you prefer deterministic ordering:
    # exp_ids.sort()

    total_time_steps = len(times)
    # experiment_data is a dict {exp_id: (T, k)}; reuse your helper
    experiment_array = prepare_experiment_array(
        total_time_steps,
        metabolite_ids,
        {eid: experiment_data[eid] for eid in exp_ids},
    )

    # 3) Dev data as array (if present)
    dev_values = list(dev_data.values())
    dev_array = np.asarray(dev_values) if len(dev_values) > 0 else None

    if verbose:
        print(f"[build_dfba_inputs] #experiments Z = {len(exp_ids)}")
        print(f"[build_dfba_inputs] #time points T = {len(times)}")
        print(f"[build_dfba_inputs] #metabolites k = {len(metabolite_ids)}")
        print(f"[build_dfba_inputs] experiment_array shape = {experiment_array.shape}")
        if dev_array is not None:
            print(f"[build_dfba_inputs] dev_array shape = {dev_array.shape}")

    return (
        times,
        metabolite_ids,
        experiment_array,
        dev_array,
        exp_ids,
        Stoichiometry,
        Transport,
        rxn_ids,
    )

def predict_dFBA(
    cobra_model_file,
    biomass_rxn_id,
    media_conc,
    ex_ids,
    time_points,
    volume=1.0,
    uptake_scaler=None,
    solver="glpk",
    mu_max=2.0,
    X0=0.01,
    X_max=10.0,
    X_init=None,
    verbose=False
):
    """
    Classical dynamic FBA predictor (single medium condition).

    Parameters
    ----------
    cobra_model_file : str
        Path to SBML/XML metabolic model.
    biomass_rxn_id : str
        ID of the biomass reaction to maximize.
    media_conc : np.array shape (k_ex,) or (1, k_ex)
        Initial extracellular metabolite concentrations, for the
        subset of metabolites corresponding to ex_ids.
        IMPORTANT: should NOT include BIOMASS.
    ex_ids : list of str
        Exchange reaction IDs corresponding to media_conc.
        (Do NOT include essential ions if you want them unconstrained.)
    time_points : np.array shape (T,)
        Simulation time grid (hours).
    volume : float
        Culture volume (L).
    uptake_scaler : array-like or None
        Per–metabolite scaling factor for uptake.
        Uptake bound is set as:
            v_max_i(t) = uptake_scaler[i] * C_i(t) / dt / volume
        If None, all ones are used.
    solver : str
        COBRA solver.
    mu_max : float
        Maximum allowed growth rate (1/hr) to avoid overflow.
    X0 : float
        Default initial biomass (gDW/L) used if X_init is None.
    X_max : float
        Maximum allowed biomass (gDW/L) to avoid overflow.
    X_init : float or None
        Experimental initial biomass (gDW/L). If not None, this
        value is used as X[0] instead of X0.

    Returns
    -------
    pred_biomass : np.array shape (T,)
        Biomass concentration predicted by dFBA at each time point.
    """

    import cobra
    import numpy as np

    # ------------------------------------------------------------
    # 0. Ensure media_conc is 1D and consistent with ex_ids
    # ------------------------------------------------------------
    media_conc = np.asarray(media_conc, dtype=float)
    if media_conc.ndim == 2 and media_conc.shape[0] == 1:
        media_conc = media_conc[0]

    if media_conc.ndim != 1:
        raise ValueError(
            f"media_conc must be 1D (k_ex,) or (1,k_ex), got shape {media_conc.shape}"
        )

    if len(ex_ids) != media_conc.shape[0]:
        raise ValueError(
            f"len(ex_ids) = {len(ex_ids)} but media_conc has length {media_conc.shape[0]}.\n"
            "They must match and correspond to the same metabolites."
        )

    # Handle uptake_scaler
    if uptake_scaler is None:
        uptake_scaler = np.ones_like(media_conc, dtype=float)
    else:
        uptake_scaler = np.asarray(uptake_scaler, dtype=float)
        if uptake_scaler.shape != media_conc.shape:
            raise ValueError(
                f"uptake_scaler shape {uptake_scaler.shape} does not match media_conc {media_conc.shape}"
            )

    # ------------------------------------------------------------
    # 1. Load metabolic model
    # ------------------------------------------------------------
    model = cobra.io.read_sbml_model(cobra_model_file)
    model.solver = solver

    if biomass_rxn_id not in model.reactions:
        raise ValueError(f"Biomass reaction {biomass_rxn_id} not found in the model.")
    biomass_rxn = model.reactions.get_by_id(biomass_rxn_id)

    # Optional: check all ex_ids exist
    for ex in ex_ids:
        if ex not in model.reactions:
            raise ValueError(f"Exchange reaction {ex} not found in the model.")

    # ------------------------------------------------------------
    # 2. Prepare time grid and initial conditions
    # ------------------------------------------------------------
    time_points = np.asarray(time_points, dtype=float)
    T = len(time_points)
    if T < 2:
        raise ValueError("time_points must contain at least 2 values.")

    dt = np.diff(time_points)
    dt = np.maximum(dt, 1e-6)  # avoid zeros/negatives

    # Biomass trajectory
    X = np.zeros(T, dtype=float)
    # Use experimental initial biomass if provided, otherwise X0
    if X_init is not None:
        X[0] = float(X_init)
    else:
        X[0] = float(X0)

    # Extracellular concentrations for the k_ex metabolites
    C = media_conc.copy()  # shape (k_ex,)

    # ------------------------------------------------------------
    # 3. dFBA integration loop
    # ------------------------------------------------------------
    for t in range(T - 1):

        dt_t = float(dt[t])

        # (a) Set uptake bounds from concentrations using scalers
        for i, ex in enumerate(ex_ids):
            rxn = model.reactions.get_by_id(ex)
            C_i = max(C[i], 0.0)

            # v_max_i = s_i * C_i / (dt * V)
            v_max_i = uptake_scaler[i] * C_i / dt_t / volume

            # For this model we assume uptake via positive flux
            # keep existing lower_bound, only tighten upper_bound
            rxn.upper_bound = v_max_i

        # (b) Solve steady-state FBA for growth
        solution = model.optimize()

        if (solution.status != "optimal") or (solution.objective_value is None):
            mu = 0.0
            uptake_fluxes = np.zeros(len(ex_ids), dtype=float)
        else:
            mu_raw = float(solution.objective_value)
            if not np.isfinite(mu_raw):
                mu = 0.0
            else:
                mu = max(0.0, min(mu_raw, mu_max))

            uptake_fluxes = np.array(
                [solution.fluxes.get(ex_id, 0.0) for ex_id in ex_ids],
                dtype=float
            )

        # (c) Update extracellular metabolites (uptake is positive flux)
        v_up = np.maximum(uptake_fluxes, 0.0)  # shape (k_ex,)

        dCdt = -v_up * X[t] / volume
        C_prev = C
        C = C + dCdt * dt_t
        C = np.maximum(C, 0.0)

        # (d) Update biomass (exponential growth)
        X_next = X[t] * np.exp(mu * dt_t)
        if not np.isfinite(X_next):
            X_next = X[t]
        X_next = min(X_next, X_max)
        X[t + 1] = X_next

        if verbose:
            print(f't={t} mu={mu} C={C_prev} dt={dt_t} X[t]={X[t]} -> X[t+1]={X_next} C[t+1]={C}')

    return X