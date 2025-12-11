# This cell is slow you should run search_parameter.py code

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sklearn
import utils
import numpy as np
import tensorflow as tf
from itertools import product
import json
from sklearn.metrics import r2_score
import random
import utils
import data
import model
import plot

# -----------------------------
# Basic config
# -----------------------------
train_test_split = 'medium'  # 'forecast' or 'medium'
folder = './'
file_name = 'putida_OD_81'
media_file = folder + 'data/' + 'putida_media_81.csv'
od_file = folder + 'data/' + 'putida_OD_81.csv'
cobra_model_file = folder + 'data/' + 'IJN1463EXP_duplicated.xml'
biomass_rxn_id = 'BIOMASS_KT2440_WT3'  # 'BIOMASS_Ec_iML1515_core_75p37M'
hidden_layers_lag = [50]
hidden_layers_flux = [500]

# Name used when saving best model
run_name = f'{file_name}_{train_test_split}'

# -----------------------------
# Parameter search spaces  (order: SV, negV, C, dropC)
# -----------------------------
l = [0.001, 0.01, 0.1, 1.0]
k = [0.0, 0.25, 0.50, 0.75, 1.0]
l1, l2, l3, l4 = l.copy(), l.copy(), l.copy(), l.copy()
k1, k2, k3, k4 = k.copy(), k.copy(), k.copy(), k.copy()
N_search = 100
param_grid = list(product(l1, l2, l4, k1, k2, k4))
random.seed(100)
param_samples = random.sample(param_grid, min(N_search, len(param_grid)))

short_num_epochs = 500
batch_size = 10
patience = 100
x_fold = 3

results = []

# Defaults (unused in search but kept for clarity)
l1_, l2_, l3_, l4_ = 1, 0.01, 1, 0.01  # order=SV,negV,C,dropC
k1_, k2_, k3_, k4_ = 0, 0.75, 0, 0

# -----------------------------
# Best model tracking
# -----------------------------
best_R2_mean = -np.inf
best_config = None

for idx, (l1_, l2_, l4_, k1_, k2_, k4_) in enumerate(param_samples):
    # Create train and val sets
    mdl, train_array, train_dev, val_array, val_dev, val_ids = model.create_model_train_val(
        media_file, od_file, cobra_model_file, biomass_rxn_id,
        x_fold=x_fold,
        hidden_layers_lag=hidden_layers_lag,
        hidden_layers_flux=hidden_layers_flux,
        lag_function='hill',
        dropout_rate=0.2,
        loss_weight=[l1_, l2_, l3_, l4_],
        loss_decay=[k1_, k2_, k3_, k4_],
        verbose=False,
        train_test_split=train_test_split
    )

    # Train
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=280, decay_rate=0.9, staircase=True
    )
    for iter in range(2):
        (
        (losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train),
        (losses_s_v_val,   losses_neg_v_val,   losses_c_val,   losses_drop_c_val)
        ) = model.train_model(
        mdl, train_array, val_array=val_array,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        num_epochs=short_num_epochs, batch_size=batch_size, patience=patience,
        verbose=False,
        train_test_split=train_test_split,
        x_fold=x_fold
        )
        
    # Predict
    pred, ref = model.predict_on_val_data(mdl, val_array, verbose=False)
    pred, ref = utils.concentration_to_OD(pred), utils.concentration_to_OD(ref)
    pred = np.expand_dims(pred, axis=0)
    ref = np.expand_dims(ref, axis=0)
    R2 = utils.r2_growth_curve(pred, ref, OD=True)
    R2_mean, R2_std, R2_median = np.mean(R2), np.std(R2), np.median(R2)

    # ---- Save best model so far ----
    if R2_mean > best_R2_mean:
        best_R2_mean = R2_mean
        best_config = (l1_, l2_, l3_, l4_, k1_, k2_, k3_, k4_)
        model_name = f'{folder}model/{run_name}_0'
        mdl.save_model(model_name=model_name, verbose=True)
        np.savetxt(f'{folder}model/{run_name}_train_array.txt', train_array, fmt='%f')
        np.savetxt(f'{folder}model/{run_name}_train_dev.txt', train_dev, fmt='%f')
        np.savetxt(f'{folder}model/{run_name}_val_array.txt', val_array, fmt='%f')
        np.savetxt(f'{folder}model/{run_name}_val_dev.txt', val_dev, fmt='%f')
        np.savetxt(f'{folder}model/{run_name}_val_ids.txt', np.asarray(val_ids), fmt='%d')
        print(
        f'Testing λ={[l1_, l2_, l3_, l4_]}, '
        f'k={[k1_, k2_, k3_, k4_]} ({idx+1}/{len(param_samples)}) '
        f'R2: {R2_mean:.3f} ± {R2_std:.3f} (median {R2_median:.3f}) BEST MODEL'
        )
    else:
        print(
        f'Testing λ={[l1_, l2_, l3_, l4_]}, '
        f'k={[k1_, k2_, k3_, k4_]} ({idx+1}/{len(param_samples)}) '
        f'R2: {R2_mean:.3f} ± {R2_std:.3f} (median {R2_median:.3f})'
        )


    # Record all results
    results.append({
        "l1": l1_,
        "l2": l2_,
        "l3": l3_,
        "l4": l4_,
        "k1": k1_,
        "k2": k2_,
        "k3": k3_,
        "k4": k4_,
        "R2_mean": R2_mean,
        "R2_std": R2_std,
        "R2_median": R2_median
    })

# Sort and print summary
results = sorted(results, key=lambda x: x["R2_median"], reverse=True)
print("\nHyperparameter Search Results order=SV,negV,C,dropC")
print("l1\tl2\tl3\tl4\tk1\tk2\tk3\tk4\tR2-mean\tR2-std\tR2-median")
for res in results:
    print(
        f"{res['l1']}\t{res['l2']}\t{res['l3']}\t{res['l4']}\t"
        f"{res['k1']}\t{res['k2']}\t{res['k3']}\t{res['k4']}\t"
        f"{res['R2_mean']:.3f}\t{res['R2_std']:.3f}\t{res['R2_median']:.3f}"
    )

