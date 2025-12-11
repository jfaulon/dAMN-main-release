# parameter-search in concentration mode

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import utils
import numpy as np
import tensorflow as tf
import random
from itertools import product
from sklearn.metrics import r2_score

# ------------------------------------------------------------------
# Training mode: concentration-ACE
#   -> train on all concentrations EXCEPT ACE in loss_c
#      and evaluate ACE as usual
# ------------------------------------------------------------------
training_mode = 'concentration-ACE'

train_test_split = utils.parse_training_mode(
    mode_str=training_mode,
    default_split="forecast",
    verbose=True
)

folder = './'
media_file       = folder + 'data/' + 'Millard_media.csv'
od_file          = folder + 'data/' + 'Millard_12_1.csv'
cobra_model_file = folder + 'data/' + 'iML1515_duplicated.xml'
biomass_rxn_id   = 'BIOMASS_Ec_iML1515_core_75p37M'

hidden_layers_lag  = [50]
hidden_layers_flux = [500]

# ------------------------------------------------------------------
# Search-space definition
# ------------------------------------------------------------------
l = [0.001, 0.01, 0.1, 1.0]
k = [0.0, 0.25, 0.5, 0.75, 1.0]

l1, l2, l3, l4 = l.copy(), l.copy(), l.copy(), l.copy()
k1, k2, k3, k4 = k.copy(), k.copy(), k.copy(), k.copy()

N_search       = 50
param_grid     = list(product(l1, l4, k2, k3))
random.seed(1)
param_samples  = random.sample(param_grid, min(N_search, len(param_grid)))

short_num_epochs = 1000
batch_size       = 5
patience         = 100
x_fold           = 1   # pour concentration-*, utils.py entraîne déjà sur tous les points de temps

results = []

# Initial values (overwritten in loop)
l1_, l2_, l3_, l4_ = 0.001, 1, 1, 1
k1_, k2_, k3_, k4_ = 0, 0.5, 0.5, 0

# ------------------------------------------------------------------
# MAIN HYPERPARAMETER SEARCH LOOP
# ------------------------------------------------------------------
for idx, (l1_, l4_, k2_, k3_) in enumerate(param_samples):

    print(f"\n============= Parameter set {idx+1}/{len(param_samples)} =============")
    print(f"Mode = {training_mode} (train_test_split='{train_test_split}')")
    print(f"λ = [{l1_}, {l2_}, {l3_}, {l4_}]   k = [{k1_}, {k2_}, {k3_}, {k4_}]")

    # --------------------------------------------------------------
    # Create training/validation sets + build model
    # --------------------------------------------------------------
    mdl, train_array, train_dev, val_array, val_dev, val_ids = model.create_model_train_val(
        media_file, od_file, cobra_model_file, biomass_rxn_id,
        x_fold=x_fold,
        hidden_layers_lag=hidden_layers_lag,
        hidden_layers_flux=hidden_layers_flux,
        dropout_rate=0.2,
        loss_weight=[k1_, l2_, l3_, l4_],
        loss_decay=[k1_, k2_, k3_, k4_],
        UB_in=0, UB_out=0.3, # ACE: UB_in=0, UB_out=0.3, GL: UB_in=1.5, UB_out=0
        verbose=False,
        train_test_split=train_test_split
    )

    metabolite_ids = list(mdl.metabolite_ids)

    # Retrieve holdout metabolite index from mdl
    holdout_idx = mdl.holdout_index
    if holdout_idx is None:
        raise RuntimeError("ERROR: concentration mode requires a holdout metabolite.")
    holdout_metabolite_id = metabolite_ids[holdout_idx]

    # --------------------------------------------------------------
    # Model training
    # --------------------------------------------------------------
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=280,
        decay_rate=0.9,
        staircase=True
    )

    (
        (losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train),
        (losses_s_v_val,   losses_neg_v_val,   losses_c_val,   losses_drop_c_val)
    ) = model.train_model(
        mdl, train_array, val_array=val_array,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        num_epochs=short_num_epochs, batch_size=batch_size,
        patience=patience, verbose=False,
        train_test_split=train_test_split, x_fold=x_fold
    )

    # --------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------
    pred, ref = model.predict_on_val_data(mdl, val_array, verbose=False)
    print("pred, ref shapes:", pred.shape, ref.shape)

    # Holdout metabolite predictions
    exp_idx   = 0
    times_vec = mdl.times
    pred_hold = pred[exp_idx, :, holdout_idx]
    ref_hold  = ref[exp_idx, :, holdout_idx]

    print("Times (validation):", times_vec)
    print(f"Predicted {holdout_metabolite_id} concentrations:", pred_hold)
    print(f"Reference {holdout_metabolite_id} concentrations:", ref_hold)

    # Expand for r2_concentrations_all
    pred = pred[None, ...]
    ref  = ref[None, ...]

    # --------------------------------------------------------------
    # R² Computation
    # --------------------------------------------------------------
    R2_mat = utils.r2_concentrations_all(pred, ref)

    R2_hold_vec    = R2_mat[:, holdout_idx]
    R2_hold_mean   = float(np.nanmean(R2_hold_vec))
    R2_hold_std    = float(np.nanstd(R2_hold_vec))
    R2_hold_median = float(np.nanmedian(R2_hold_vec))

    R2_all_mean   = float(np.nanmean(R2_mat))
    R2_all_std    = float(np.nanstd(R2_mat))
    R2_all_median = float(np.nanmedian(R2_mat))

    # --------------------------------------------------------------
    # Print R² *per metabolite*
    # --------------------------------------------------------------
    print(f"    R² per metabolite:")
    for j, met in enumerate(metabolite_ids):
        vals = R2_mat[:, j]   # all experiments for metabolite j
        print(f"        {met:12s}: mean={np.nanmean(vals):.3f}, std={np.nanstd(vals):.3f}, "
              f"median={np.nanmedian(vals):.3f}")
    print(f"    Global R²: mean = {R2_all_mean:.3f} ± {R2_all_std:.3f}")

    results.append({
        "l1": l1_, "l2": l2_, "l3": l3_, "l4": l4_,
        "k1": k1_, "k2": k2_, "k3": k3_, "k4": k4_,
        "R2_hold_mean":   R2_hold_mean,
        "R2_hold_std":    R2_hold_std,
        "R2_hold_median": R2_hold_median,
        "R2_all_mean":    R2_all_mean,
        "R2_all_std":     R2_all_std,
        "R2_all_median":  R2_all_median
    })


# ------------------------------------------------------------------
# Sort and print by GLOBAL R² (R2_all_mean)
# ------------------------------------------------------------------
results_all_sorted = sorted(results, key=lambda x: x["R2_all_mean"], reverse=True)

print(f"\nHyperparameter Search Results — SORTED BY GLOBAL R² (R2_all_mean)")
print("l1\tl2\tl3\tl4\tk1\tk2\tk3\tk4\tR2_all_mean\tR2_all_std\tR2_all_median\tR2_hold_mean")

for res in results_all_sorted:
    print(
        f"{res['l1']}\t{res['l2']}\t{res['l3']}\t{res['l4']}\t"
        f"{res['k1']}\t{res['k2']}\t{res['k3']}\t{res['k4']}\t"
        f"{res['R2_all_mean']:.3f}\t{res['R2_all_std']:.3f}\t"
        f"{res['R2_all_median']:.3f}\t{res['R2_hold_mean']:.3f}"
    )