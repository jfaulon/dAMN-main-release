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
# PLOTTING UTILS
###############################################################################

def plot_loss(title, loss, num_epochs, save=''):
    """
    Generate a high-quality scientific plot for a single loss curve over training epochs.
    Parameters:
        title (str): Title of the plot.
        loss (array-like): Loss values (e.g., total loss or any individual loss).
        num_epochs (int): Number of epochs.
        save (str): Folder where the plot should be saved (as a PNG). If empty, plot is not saved.
    """
    
    # Set up scientific figure style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log(Loss)', color='blue')
    ax.plot(range(1, len(loss) + 1), loss, label='Loss', color='blue')
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelcolor='blue')

    plt.title(f'Loss over {num_epochs} Epochs ({title})', pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    plt.tight_layout()
    # Save high-resolution figure if save path is provided
    if save != '':
        safe_title = title.replace(' ', '_')
        plt.savefig(f'{save}/loss_plot_{safe_title}.png', dpi=300, bbox_inches='tight')

def plot_predicted_reference_growth_curve(
    times, Pred, Ref,
    val_dev=None,
    OD=True, R2=None, R2dev=None,
    train_time_steps=0,
    experiment_ids=None,
    run_name="run",
    train_test_split="medium",
    save=None,
    R2min=0, R2max=1,
):
    """
    Plot all growth curves (OD or concentration) for multiple experiments/iterations.
    If train_time_steps is provided, splits the predicted curve at that point
    and draws a vertical line.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    N_iter, N_exp, n_times, n_met = Pred.shape

    Pred_bio = Pred[..., -1]
    Ref_bio = Ref[..., -1]

    if OD:
        Pred_bio, Ref_bio = utils.concentration_to_logOD(Pred_bio), utils.concentration_to_logOD(Ref_bio)
        ylabel = "log(OD)"
    else:
        ylabel = "Concentration (mM)"

    pred_bio_mean, ref_bio_mean = np.mean(Pred_bio, axis=0), np.mean(Ref_bio, axis=0)
    pred_bio_std, ref_bio_std   = np.std(Pred_bio, axis=0), (val_dev if val_dev is not None else np.std(Ref_bio, axis=0))

    for i in range(N_exp):
        if R2 is not None:
            if not (R2min <= R2[i] <= R2max):
                continue
                
        exp_label = f"Experiment {experiment_ids[i]}" if experiment_ids is not None else f"Exp {i}"
        title = f"{exp_label} {train_test_split}"

        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 16,
            'axes.labelsize': 18,
            'axes.titlesize': 18,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })

        # Plot reference
        plt.plot(times, ref_bio_mean[i], marker='o', linestyle='-', color='black', label='Reference')

        # Plot prediction: split at train_time_steps if needed
        if 0 < train_time_steps < len(times):
            # Prediction up to forecast boundary
            plt.plot(times[:train_time_steps+1], pred_bio_mean[i][:train_time_steps+1],
                     marker='s', linestyle='--', color='darkgreen', label='Prediction')
            # Forecast region
            plt.plot(times[train_time_steps:], pred_bio_mean[i][train_time_steps:],
                     marker='s', linestyle='--', color='darkgreen')
            # Connector
            plt.plot(
                [times[train_time_steps-1], times[train_time_steps]],
                [pred_bio_mean[i][train_time_steps-1], pred_bio_mean[i][train_time_steps]],
                linestyle='-', color='darkgreen', linewidth=3, alpha=0.6
            )
            plt.axvline(times[train_time_steps], color='k', 
                        linestyle=':', alpha=0.6, label='Forecast Start')
        else:
            plt.plot(times, pred_bio_mean[i], marker='s', 
                     linestyle='--', color='darkgreen', label='Prediction')

        if ref_bio_std is not None:
            plt.fill_between(times, ref_bio_mean[i] - ref_bio_std[i], 
                             ref_bio_mean[i] + ref_bio_std[i], color='black', alpha=0.2)
        if pred_bio_std is not None:
            plt.fill_between(times, pred_bio_mean[i] - pred_bio_std[i], 
                             pred_bio_mean[i] + pred_bio_std[i], color='darkgreen', alpha=0.2)

        plt.xlabel('Time (h)')
        plt.ylabel(ylabel)
        plt.title(title, pad=20)
        
        # ---- Print R2 on plot ----
        if R2 is not None:
            R2text =  f"R²: {R2[i]:.2f}±{R2dev[i]:.2f}" if R2dev is not None else f"R²: {R2[i]:.2f}"
            plt.text(0.05, 0.95, R2text, transform=plt.gca().transAxes,
                     fontsize=14, verticalalignment='top', 
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
         
        plt.grid(True, linestyle='--', alpha=0.6)
        #plt.legend(frameon=True, loc='best', fancybox=True)
        plt.tight_layout()

        if save:
            title_clean = f"{title.replace(' ', '_').replace('/', '')}"
            plt.savefig(f'{save}/{title_clean}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_predicted_biomass_and_substrate(
    times, Pred,
    experiment_ids=None,
    metabolite_ids=None,
    run_name="run",
    train_test_split="medium",
    save=None
):
    """
    For each experiment, plot (on the same figure) the predicted growth curve (biomass, gDW/L) and
    the predicted substrate concentration (mM, substrate index is detected per experiment).
    Only Predicted values are plotted.
    Pred: (N_iter, N_exp, n_times, n_met)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    N_iter, N_exp, n_times, n_met = Pred.shape

    # --- Find substrate indices ---
    def substrate_id(data):
        valid_mask = (~np.isnan(data)) & (data != 0)
        first_valid_indices = np.full((data.shape[0], data.shape[1]), -1)
        for i in range(data.shape[0]):
            for t in range(data.shape[1]):
                valid = valid_mask[i, t]
                if np.any(valid):
                    first_valid_indices[i, t] = np.argmax(valid)
        id = []
        for i in range(data.shape[0]):
            found = first_valid_indices[i][first_valid_indices[i] != -1]
            id.append(found[0] if len(found) else -1)
        return id

    sub_ids = substrate_id(Pred[0])   # shape (N_exp,)

    # --- Extract BIOMASS and SUBSTRATE curves ---
    Pred_bio = Pred[..., -1]
    Pred_sub = np.zeros((N_iter, N_exp, n_times))
    for i in range(N_exp):
        idx = sub_ids[i]
        if idx < 0:
            Pred_sub[:, i, :] = np.nan
        else:
            Pred_sub[:, i, :] = Pred[:, i, :, idx]

    # --- Mean and std across N_iter (axis=0) ---
    pred_bio_mean = np.mean(Pred_bio, axis=0)
    pred_bio_std = np.std(Pred_bio, axis=0)
    pred_sub_mean = np.mean(Pred_sub, axis=0)
    pred_sub_std = np.std(Pred_sub, axis=0)

    # --- Plot ---
    for i in range(N_exp):
        exp_label = f"Experiment {experiment_ids[i]}" if experiment_ids is not None else f"Exp {i}"
        sub_name = metabolite_ids[sub_ids[i]] if (metabolite_ids is not None and sub_ids[i] >= 0) else "Unknown"
        title = f"{exp_label} {train_test_split} {sub_name}"
        print(f"Experiment {exp_label}: Substrate = {sub_name} (index {sub_ids[i]})")

        # 1. Use subplots and twinx for dual y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 16,
            'axes.labelsize': 18,
            'axes.titlesize': 18,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })

        # 2. Plot biomass (left axis)
        ax1.plot(times, pred_bio_mean[i], marker='s', linestyle='--', color='darkgreen', label='Pred Biomass')
        ax1.fill_between(times, pred_bio_mean[i] - pred_bio_std[i], pred_bio_mean[i] + pred_bio_std[i], color='darkgreen', alpha=0.2)
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel("Biomass (gDW/L)", color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # 3. Plot substrate (right axis)
        ax2 = ax1.twinx()
        ax2.plot(times, pred_sub_mean[i], marker='v', linestyle='--', color='orange', label=f'Pred Substrate ({sub_name})')
        ax2.fill_between(times, pred_sub_mean[i] - pred_sub_std[i], pred_sub_mean[i] + pred_sub_std[i], color='orange', alpha=0.15)
        ax2.set_ylabel(f"{sub_name} (mM)", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # 4. Title and grid (on ax1)
        # plt.title(title, pad=20)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 5. Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines1 + lines2, labels1 + labels2, frameon=True, loc='best', fancybox=True)

        plt.tight_layout()

        if save:
            title_clean = f"{title.replace(' ', '_').replace('/', '')}"
            plt.savefig(f'{save}/{title_clean}.png', dpi=300, bbox_inches='tight')
        plt.show()

    times=times,
    time_cutoffs=9,   # keep this only if your function signature supports it
    Ref=Ref,
    Pred_mean=Pred_mean,
    Pred_std=Pred_std,
    metabolite_ids=metabolite_ids,
    val_ids=val_ids,
    run_name=run_name,
    N_models=N_models,
    R2_mat=R2_mat,
    save_dir="./figure"

def plot_ensemble_concentrations(
    times=[],
    metabolite_ids=[],
    Ref=[],
    Pred_mean=[],
    Pred_std=[],
    time_cutoffs=0,
    R2_mat=None,
    val_ids=None,
    run_name="ensemble_conc",
    N_models=3,
    save_dir=None,
):
    """
    Plot ensemble mean ± std for each metabolite and experiment.
    IMPORTANT: 
      - NO subplot grid: each figure has a SINGLE Axes.
      - Each figure has figsize = (10, 6).
      - One figure per (experiment, metabolite).
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    times = np.asarray(times)
    Ref = np.asarray(Ref)
    Pred_mean = np.asarray(Pred_mean)
    Pred_std = np.asarray(Pred_std)

    if time_cutoffs > 0:
        idx = times <= time_cutoffs
        times = times[idx]
        Ref = Ref[:, idx, :]
        Pred_mean = Pred_mean[:, idx, :]
        Pred_std  = Pred_std[:, idx, :]
    
    Z, T, k_met = Pred_mean.shape

    if val_ids is None:
        val_ids = np.arange(Z)

    # Style to match plot_predicted_biomass_and_substrate
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    })

    for exp_idx, exp_id in enumerate(val_ids):
        for j, met in enumerate(metabolite_ids):

            y_ref  = Ref[exp_idx, :, j]
            y_mean = Pred_mean[exp_idx, :, j]
            y_std  = Pred_std[exp_idx, :, j]

            fig, ax = plt.subplots(figsize=(8, 6), dpi=300) 

            # Reference data: dots
            ax.plot(times, y_ref, "o", color="black", label="Ref")

            # Mean prediction: line
            ax.plot(times, y_mean, "-", color="tab:blue", label="Pred mean")

            # Shaded ±1 std
            ax.fill_between(
                times,
                y_mean - y_std,
                y_mean + y_std,
                color="tab:blue",
                alpha=0.2,
                label="±1 std",
            )

            # Title with optional R²
            if R2_mat is not None:
                r2_val = R2_mat[exp_idx, j]
                if not np.isnan(r2_val):
                    title = f"{met} (R²={r2_val:.2f})"
                else:
                    title = met
            else:
                title = met
            ax.set_title(title)

            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.grid(True, linestyle="--", alpha=0.3)

            # Legend
            # ax.legend(loc="upper right")

            # Global title
            fig.suptitle(
                f"{run_name} – exp {int(exp_id)} – {met} – ensemble of {N_models} models",
                fontsize=18,
            )

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            if save_dir:
                fig_path = os.path.join( save_dir, f"{run_name}_exp{int(exp_id)}_{met}_ensemble.png")
                fig.savefig(fig_path, dpi=300)
            plt.show()
            plt.close(fig)
        
def plot_similarity_distribution(title, r2_values, save=''):
    """
    Plot R2 value distribution
    """
    r2_values = np.asarray(r2_values)
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'lines.linewidth': 2
    })

    # ---- Set bins with fixed width 0.05
    bin_width = 0.1
    min_val = np.floor(r2_values.min() / bin_width) * bin_width
    max_val = np.ceil(r2_values.max() / bin_width) * bin_width
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    n, bins, patches = plt.hist(r2_values, bins=bins, color='grey', edgecolor='black', alpha=0.7, density=True)
    
    # Convert density to percent
    bin_widths = np.diff(bins)
    n_percent = n * bin_widths * 100  # now n_percent sums to 100%
    plt.clf()  # Clear to redraw

    # Plot again with percent
    plt.bar(bins[:-1], n_percent, width=bin_widths, color='grey', edgecolor='black', alpha=0.7, align='edge')
    plt.xlabel('R2')
    plt.ylabel('Frequency (%)')
    #plt.title(f"{title}\nR2={mean_r2:.2f}±{std_r2:.2f}", pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save != '':
        safe_title = title.replace(' ', '_')
        plt.savefig(f'{save}/{safe_title}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
