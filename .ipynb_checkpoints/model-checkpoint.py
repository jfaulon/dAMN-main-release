import os
from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf
import pandas as pd
import cobra
import json
import matplotlib.pyplot as plt
import sklearn
import utils
import data

###############################################################################
# CREATE MODELS
###############################################################################

class LagNetwork(tf.keras.layers.Layer):
    """
    Neural network for learning lag-phase parameters from initial conditions.
    This layer predicts two parameters for each trajectory:
      - t_lag:   lag time (duration of lag phase)
      - r_lag:   stiffness parameter for the sigmoid ramp
    """
    def __init__(self, input_dim, hidden_layers=[50], dropout_rate=0.2, name='lag_network', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(h, activation='relu', name=f'lag_hidden_{i}')
            for i, h in enumerate(hidden_layers)
        ])
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name='lag_dropout')
        self.out_layer = tf.keras.layers.Dense(2, activation=None, name='lag_output')  # 2 outputs: t_lag and r_lag

    def call(self, x, training=False):
        x = self.hidden_layers(x)
        x = self.dropout(x, training=training)
        out = self.out_layer(x)
        # For stability: softplus for r_lag (so it's positive), softplus or relu for t_lag (positive)
        t_lag = tf.nn.softplus(out[..., 0:1])  # shape (batch, 1)
        r_lag = tf.nn.softplus(out[..., 1:2])  # shape (batch, 1)
        return t_lag, r_lag

class FluxNetwork(tf.keras.layers.Layer):
    """
    Neural network for learning fluxes from concentration
    """
    def __init__(self, input_dim, output_dim, hidden_layers=[500], dropout_rate=0.2, name='flux_network', **kwargs):
        super().__init__(name=name, **kwargs)
        # Use Sequential to ensure Keras tracks all sublayers
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=h, 
                activation='relu', 
                name=f'hidden_dense_{i}'
            ) for i, h in enumerate(hidden_layers)
        ], name='hidden_layers')

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.out_layer = tf.keras.layers.Dense(
            units=output_dim, 
            activation='linear', 
            name='output_dense'
        )

    def call(self, x, training=False):
        x = self.hidden_layers(x)
        x = self.dropout(x, training=training)
        return self.out_layer(x)

class MetabolicModel(tf.keras.Model):
    """
    Model that unrolls the time dimension.
    Expects input shape: (batch_size, T, k)
    dt_vector: tensor of shape (T,) with time differences (t[i+1]-t[i])
    """

    def __init__(self, times, metabolite_ids, Transport, Stoichiometry, rxn_ids, biomass_rxn_id, 
        hidden_layers_lag=[0], hidden_layers_flux=[500], dropout_rate=0.2, 
        loss_weight=[0, 0, 0, 0], loss_decay=[0, 0, 0, 0],
        UB_in=0.0, # UB_in : max allowed consumption per step  (C_pred - C_next)
        UB_out=0.0, # UB_out: max allowed production per step   (C_next - C_pred)
        lag_function='exponential',
        train_test_split='medium', x_fold=5,
        train_time_steps=0,
        verbose=True, name='metabolic_model'):
        
        super().__init__(name=name)
        self.times = times
        self.metabolite_ids = metabolite_ids
        self.T = len(times)  # number of time points
        self.train_time_steps = train_time_steps if train_time_steps > 0 else self.T

        # Normalized time-step vector (length T-1 if times has length T)
        dt_vector = np.diff(times).astype(np.float32)
        dt_vector = dt_vector / np.min(dt_vector)
        self.dt_vector = tf.convert_to_tensor(dt_vector, dtype=tf.float32)
        self.Transport = Transport
        self.Stoichiometry = Stoichiometry
        self.k, self.n = Transport.shape[0], Transport.shape[1]
        self.rxn_ids = rxn_ids
        self.biomass_rxn_id = biomass_rxn_id
        self.biomass_flux_index = rxn_ids.index(biomass_rxn_id)
        self.hidden_layers_lag = hidden_layers_lag
        self.hidden_layers_flux = hidden_layers_flux
        self.UB_in = float(UB_in) if UB_in is not None else 0.0
        self.UB_out = float(UB_out) if UB_out is not None else 0.0
        if self.UB_in > 0.0 and self.UB_out > 0.0:
            raise ValueError(
                "UB_in and UB_out cannot both be positive: "
                "the holdout metabolite is either consumed (UB_in>0) "
                "or produced (UB_out>0), but not both."
            )
        self.lag_function = lag_function
        if self.lag_function not in ("exponential", "hill"):
            raise ValueError(
                f"Unknown lag_function '{self.lag_function}'. "
                "Valid options are 'exponential' and 'hill'."
            )      
        self.dropout_rate = dropout_rate
        self.loss_weight = loss_weight
        self.loss_decay = loss_decay
        self.train_test_split = train_test_split
        self.x_fold = x_fold

        # Index of metabolite (if any) held out from concentration loss.
        # This is set externally (e.g. via data.get_holdout_index_from_split()).
        self.holdout_index = None

        # Add lag network: maps initial condition to (t_lag, r_lag)
        if self.hidden_layers_lag[0] > 0:
            self.lag_net = LagNetwork(
                self.k,
                hidden_layers=hidden_layers_lag,
                dropout_rate=dropout_rate,
                name='lag_network'
            )

        # Initialize FluxNetwork  
        self.flux_net = FluxNetwork(
            self.k,
            self.n,
            hidden_layers_flux,
            dropout_rate,
            name='flux_network'
        )      

    def printout(self):
        print(f'-----------------------------MetabolicModel-----------------------------')
        print(f'times: {self.times[0]:.2f}, {self.times[1]:.2f}, ..., {self.times[-1]:.2f}')
        print(f'metabolite_ids: {self.metabolite_ids}')
        print(f'Total time step: {self.T}')
        print(f'Train time step: {self.train_time_steps}')
        print(f'dt: {self.dt_vector.numpy()}')
        print(f'Transport: {self.Transport.shape}')
        print(f'Stoichiometry: {self.Stoichiometry.shape}')
        print(f'n: {self.n}')
        print(f'k: {self.k}')
        print(f'Reaction ids: {len(self.rxn_ids)}')
        print(f'Biomass id: {self.biomass_rxn_id}')
        print(f'Biomass flux index: {self.biomass_flux_index}')
        if  self.hidden_layers_lag[0] > 0:
            print(f'Lag Layer : Hidden size = {self.hidden_layers_lag} trainainle parameters = {self.lag_net.count_params()}')
        print(f'Flux Layer : Hidden size = {self.hidden_layers_flux} trainainle parameters = {self.flux_net.count_params()}')
        print(f'Upper bounds for in/out medium metaboilites: {self.UB_in, self.UB_out}')
        print(f'Lag function     : {self.lag_function}')
        print(f'Dropout Rate: {self.dropout_rate}')
        print(f'Loss weight: {self.loss_weight}')
        print(f'Loss decay: {self.loss_decay}')
        print(f'train_test_split: {getattr(self, "train_test_split", "N/A")}')
        print(f'train_test_split: {getattr(self, "train_time_steps", "N/A")}')
        print(f'x_fold: {getattr(self, "x_fold", "N/A")}')
        print(f'------------------------------------------------------------------------')

    def debug_concentration_step(self, t, C_pred, C_next, delta_C, v_t, lag_params, rt, verbose):
        """Prints debug information for concentration updates."""
        """we print experiment I == verbose-1"""

        if len(self.metabolite_ids) == 0:
            return
        I = verbose-1 # expeimrnat to be printed
        tf.print(f't={t}------------------------------------------------------------------')
        t_lag = lag_params['t_lag']
        r_lag = lag_params['r_lag']
        t_lag = tf.reshape(t_lag, [-1, 1])
        r_lag = tf.reshape(r_lag, [-1, 1])    
        print(f"Experiment {I+1}: t_lag = {t_lag[I, 0].numpy():.4f}, r_lag = {r_lag[I, 0].numpy():.4f} r_t={float(rt[I])}")

        # Print non-zero C_pred (for t=0)
        if float(t) == 0.0 and self.metabolite_ids is not None:
            nonzero_mask_pred = tf.not_equal(C_pred[I], 0.0)
            nonzero_indices_pred = tf.where(nonzero_mask_pred)
            nonzero_values_pred = tf.gather(C_pred[I], nonzero_indices_pred)
            tf.print("C_pred non-zero (t=0):")
            for idx, val in zip(tf.reshape(nonzero_indices_pred, [-1]).numpy(), tf.reshape(nonzero_values_pred, [-1]).numpy()):
                met_name = self.metabolite_ids[idx] if idx < len(self.metabolite_ids) else f"Met{idx}"
                print(f'  idx={idx}  {met_name} = {val:.2f}')

        # print v_t statistics
        tf.print(f'v_t stats min: {tf.reduce_min(v_t[I]):.2f} ' \
                 f'max: {tf.reduce_max(v_t[I]):.2f} ' \
                 f'mean : {tf.reduce_mean(v_t[I]):.2f} ' \
                 f'num NaN: {tf.reduce_sum(tf.cast(tf.math.is_nan(v_t[I]), tf.int32))}')

        # Print non-zero delta_C C_next
        nonzero_mask_next = tf.not_equal(C_next[I], 0.0)
        nonzero_indices_next = tf.where(nonzero_mask_next)
        nonzero_values_next = tf.gather(C_next[I], nonzero_indices_next)
        nonzero_values_delta = tf.gather(delta_C[I], nonzero_indices_next)
        tf.print("delta_C non-zero:")
        for idx, val in zip(tf.reshape(nonzero_indices_next, [-1]).numpy(), tf.reshape(nonzero_values_delta, [-1]).numpy()):
            met_name = self.metabolite_ids[idx] if idx < len(self.metabolite_ids) else f"Met{idx}"
            print(f'  idx={idx}  {met_name} = {val:.2f}')
        tf.print("C_next non-zero:")
        for idx, val in zip(tf.reshape(nonzero_indices_next, [-1]).numpy(), tf.reshape(nonzero_values_next, [-1]).numpy()):
            met_name = self.metabolite_ids[idx] if idx < len(self.metabolite_ids) else f"Met{idx}"
            print(f'  idx={idx}  {met_name} = {val:.2f}')

    def _compute_lag_params(self, C_pred, t, training, lag_params=None, verbose=False):
        """
        Compute r_t_broadcast and lag_params.
        - If hidden_layers_lag[0] <= 0, returns r_t = 1 (no lag).
        - Otherwise, uses self.lag_net to get t_lag and r_lag, and
          then computes r_t according to self.lag_function:
            * "exponential" : legacy behavior (your original code)
            * "hill"        : Hill function with n=4 and t_lag
        Returns
        -------
        r_t_broadcast : tf.Tensor, shape (batch_size, 1) or scalar 1.0
        lag_params    : dict or None
        """
        # If no lag network, just return r_t = 1
        if self.hidden_layers_lag[0] <= 0:
            return 1.0, None

        # --- 1. Get or reuse lag parameters from lag_net -----------------
        if lag_params is None:
            t_lag, r_lag = self.lag_net(C_pred, training=training)
            lag_params = dict(t_lag=t_lag, r_lag=r_lag)
        else:
            t_lag = lag_params['t_lag']
            r_lag = lag_params['r_lag']

        # Ensure shape: (batch_size, 1)
        t_lag = tf.reshape(t_lag, [-1, 1])
        r_lag = tf.reshape(r_lag, [-1, 1])

        # t may be scalar or tensor; make compatible with batch
        if not tf.is_tensor(t):
            t = tf.convert_to_tensor(t, dtype=tf.float32)
        t = tf.reshape(t, [1, 1])                  # (1, 1)
        t = tf.broadcast_to(t, tf.shape(t_lag))    # (batch_size, 1)

        # --- 2. Compute r_t according to lag_function --------------------
        eps = 1e-8

        if self.lag_function == "exponential":
            # r(t) = (1 - exp(-r_lag * t)) / (1 - exp(-r_lag * t_lag))
            numerator = 1.0 - tf.exp(-r_lag * t)
            denominator = 1.0 - tf.exp(-r_lag * t_lag)
            # Prevent division by zero (when r_lag * t_lag is very small)
            denominator = tf.where(
                tf.abs(denominator) < eps,
                tf.ones_like(denominator),
                denominator
            )
            r_t_exp = numerator / denominator
            # Before t_lag: follow exponential; after t_lag: clamp to 1
            r_t = tf.where(t < t_lag, r_t_exp, tf.ones_like(r_t_exp))
            r_t = tf.clip_by_value(r_t, 0.0, 1.0)
        elif self.lag_function == "hill":
            # Hill-type with fixed exponent n = 4:
            # r(t) = (t / t_lag)^n / (1 + (t / t_lag)^n)
            #  -> ~0 when t << t_lag, ~1 when t >> t_lag
            n = 4.0
            # Ensure positivity and avoid division by zero
            t_pos = tf.nn.relu(t)
            t_lag_pos = tf.nn.relu(t_lag) + eps
            ratio = t_pos / t_lag_pos
            ratio_pow = tf.pow(ratio, n)
            r_t = ratio_pow / (1.0 + ratio_pow)
            r_t = tf.clip_by_value(r_t, 0.0, 1.0)
        else:
            raise ValueError(
                f"Unknown lag_function '{self.lag_function}'. "
                "Valid options are 'exponential' and 'hill'."
            )

        r_t_broadcast = r_t  # shape (batch_size, 1)
        return r_t_broadcast, lag_params
    
    def _next_concentration(self, C_pred, t, training, lag_params=None, verbose=False):
        """
        Compute the next concentration vector, including lag-phase, for any time step.
        Enforces non-negativity for extracellular metabolites (*_e).
        """

        # 1. Compute lag-phase parameters
        r_t, lag_params = self._compute_lag_params(
            C_pred, t, training, lag_params, verbose=verbose
        )

        # 2. Predict fluxes
        v_t = self.flux_net(C_pred, training=training)

        # 3. Compute dC/dt = v_t @ Transport^T
        delta_C = tf.matmul(v_t, self.Transport, transpose_b=True)

        # 4. Time-step size (normalized dt)
        t_idx = int(t) if isinstance(t, (int, np.integer)) else int(t.numpy())
        dt = self.dt_vector[-1] if t_idx >= self.dt_vector.shape[0] else self.dt_vector[t_idx]

        # 5. Next concentration update 
        C_next = C_pred + r_t * delta_C * dt
    
        # 6. no negative concentrations
        if getattr(self, "holdout_index", None) is not None: 
            ext_mask = tf.constant(
            [met.endswith("_e") for met in self.metabolite_ids],
            dtype=tf.bool
            )
            # Broadcast mask to batch
            mask_b = tf.broadcast_to(ext_mask, tf.shape(C_next))
            # Apply softplus only to extracellular metabolites
            C_next = tf.where(mask_b, tf.nn.softplus(C_next), C_next)

        # Debug
        if verbose:
            self.debug_concentration_step(
                t, C_pred, C_next, delta_C, v_t, lag_params, r_t, verbose
            )

        return C_next, v_t, lag_params
        
    def compute_losses(self, v_t, C_pred, C_next, C_ref_next, loss_weight, decay):
        """
        Compute the 4 loss components given the current state.
        """

        if getattr(self, "holdout_index", None) is not None:
            idx = self.holdout_index        # concentration mode: holdout metabolite
        else:
            idx = -1                        # default → biomass (last column)

        # 1) Stoichiometric Violation Loss (SV)
        loss = tf.matmul(v_t, self.Stoichiometry, transpose_b=True)
        loss_s_v = loss_weight[0] * decay[0] * tf.reduce_mean(tf.square(loss))

        # 2) Negative Flux Loss
        loss = tf.reduce_mean(tf.nn.relu(-v_t))
        loss_neg_v = loss_weight[1] * decay[1] * loss

        # 3) Concentration Loss (drop holdout metabolite from loss_c)
        mask = ~tf.math.is_nan(C_ref_next)    # NaN mask 
        if idx != -1:  # If we have a holdout metabolite, DROP it from loss_c
            k = tf.shape(mask)[1]
            # boolean mask for columns (True = keep, False = drop holdout)
            col_mask = tf.ones([k], dtype=tf.bool)
            col_mask = tf.tensor_scatter_nd_update(
                col_mask,
                indices=[[idx]],
                updates=[False]
            )
            col_mask = tf.reshape(col_mask, [1, -1])
            col_mask = tf.broadcast_to(col_mask, tf.shape(mask))
            mask = tf.logical_and(mask, col_mask)
        
        diff = tf.boolean_mask(C_next,     mask) \
             - tf.boolean_mask(C_ref_next, mask)
        loss_c = loss_weight[2] * decay[2] * tf.reduce_mean(tf.square(diff))

        # 4) Drop-Loss
        #    - If idx == -1: keep original "biomass should not drop" penalty
        #    - If idx != -1: constrain per-step change of the holdout metabolite
        if idx == -1:
            # Default: penalize drop in biomass concentration (last metabolite)
            drop_term = tf.nn.relu(C_pred[:,-1]-C_next[:,-1])
        else:
            # Holdout metabolite
            c_pred = C_pred[:, idx]
            c_next = C_next[:, idx]
            # Positive = consumption (concentration decreases)
            consumption = c_pred - c_next
            # Positive = production (concentration increases)
            production = c_next - c_pred
            drop_terms = []
            # UB_in: max allowed consumption per time step
            if self.UB_in > 0.0:
                # Penalize if consumption exceeds UB_in
                excess_consumption = tf.nn.relu(consumption - self.UB_in)
                drop_terms.append(excess_consumption)
            # UB_out: max allowed production per time step
            if self.UB_out > 0.0:
                # Penalize if production exceeds UB_out
                excess_production = tf.nn.relu(production - self.UB_out)
                drop_terms.append(excess_production)
            if drop_terms:
                drop_term = tf.add_n(drop_terms)
            else:
                # No UB constraints: no drop_c penalty on holdout metabolite
                drop_term = tf.zeros_like(c_pred)

        loss_drop_c = loss_weight[3] * decay[3] * tf.reduce_mean(drop_term)
        
        return loss_s_v, loss_neg_v, loss_c, loss_drop_c
        
    def call(self, C_ref_batch, training=False, verbose=False):
        """
        C_ref_batch: shape (batch_size, T+1, k)
        dt_vector: shape (T,) containing dt for each time interval.
        """
        batch_size = tf.shape(C_ref_batch)[0]
        T = tf.shape(C_ref_batch)[1]
        t_max = tf.cast(T, tf.float32)
        C0 = C_ref_batch[:, 0, :]   # shape (batch_size, k)
        lag_params = None
        C_pred = C_ref_batch[:, 0, :]  # shape: (batch_size, k)
        loss_s_v, loss_neg_v, loss_c, loss_drop_c  = 0.0, 0.0, 0.0, 0.0
        verbose = False if training else verbose
        
        for t in tf.range(T-1):

            ttf = tf.cast(t, tf.float32)

            C_next, v_t, lag_params = self._next_concentration(
            C_pred, ttf, training=training, lag_params=lag_params, verbose=verbose
            )
 
            # Reference concentrations for time t+1, shape: (batch_size, k)
            C_ref_next = C_ref_batch[:, t+1, :]
            
            # Compute all 4 losses at this step
            decay = [tf.exp(-self.loss_decay[i] * ttf) for i in range(4)]
            ls_v, lnv, lc, ldc = self.compute_losses(
            v_t, C_pred, C_next, C_ref_next,  self.loss_weight, decay
            )
            loss_s_v += ls_v
            loss_neg_v += lnv
            loss_c += lc
            loss_drop_c += ldc

            C_pred = C_next

        return loss_s_v, loss_neg_v, loss_c, loss_drop_c
        
    def save_model(self, model_name='dAMNmodel', verbose=False):
        """
        Save the model weights and configuration (architecture parameters) to files.
        """
        weights_path = f'{model_name}.weights.h5'
        config_path = f'{model_name}.config.json'

        # Ensure model is built before saving by running a dummy forward pass
        dummy_input = tf.random.normal((1, len(self.times), self.k))  # Shape: (batch_size, T+1, k)
        _ = self(dummy_input)  # Forces all layers to initialize

        # Save the weights
        self.save_weights(weights_path)

        # Save model configuration
        config = {
        'times': self.times.tolist() if isinstance(self.times, (np.ndarray, list)) else list(self.times),
        'train_time_steps': int(self.train_time_steps),
        'metabolite_ids': list(self.metabolite_ids),
        'Transport': self.Transport.tolist(),
        'Stoichiometry': self.Stoichiometry.tolist(),
        'k': int(self.k),
        'n': int(self.n),
        'rxn_ids': self.rxn_ids,
        'biomass_rxn_id': self.biomass_rxn_id,
        'biomass_flux_index': int(self.biomass_flux_index),
        'hidden_layers_lag': self.hidden_layers_lag,
        'hidden_layers_flux': self.hidden_layers_flux,
        'UB_in': float(getattr(self, "UB_in", 0.0)),
        'UB_out': float(getattr(self, "UB_out", 0.0)),
        'lag_function': self.lag_function,         
        'dropout_rate': float(self.dropout_rate),
        'loss_weight': self.loss_weight,
        'loss_decay': self.loss_decay,
        'train_test_split': getattr(self, "train_test_split", "medium"),
        'x_fold': getattr(self, "x_fold", 5),
        }
    
        # Write config to JSON file
        with open(config_path, 'w') as f:
            json.dump(config, f)

        if verbose:
            print(f'Model weights saved to {weights_path} and config saved to {config_path}')

    @classmethod
    def load_model(cls, model_name='dAMNmodel', metabolite_ids=[], verbose=False):
        """
        Load a model from saved weights and configuration.
        """
        weights_path = f'{model_name}.weights.h5'
        config_path = f'{model_name}.config.json'

        # Check if files exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Configuration file {config_path} not found.')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'Weights file {weights_path} not found.')

        # Load the configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Extract parameters
        times = config['times']
        T = len(times)
        train_time_steps = config.get('train_time_steps', T)
        metabolite_ids = list(config.get('metabolite_ids', metabolite_ids))
        Transport = np.array(config['Transport'])
        Stoichiometry = np.array(config['Stoichiometry'])
        rxn_ids = config['rxn_ids']
        biomass_rxn_id = config['biomass_rxn_id']
        hidden_layers_lag = config['hidden_layers_lag']
        hidden_layers_flux = config['hidden_layers_flux']
        dropout_rate = config['dropout_rate']
        loss_weight = config['loss_weight']
        loss_decay = config['loss_decay']
        train_test_split = config.get('train_test_split', 'medium')
        x_fold = config.get('x_fold', 5)
        UB_in = float(config.get('UB_in', 0.0))
        UB_out = float(config.get('UB_out', 0.0))
        lag_function = config.get('lag_function', 'exponential')       
        
        # Create a new model instance
        model = cls(
        times=times,
        metabolite_ids=metabolite_ids,
        train_time_steps=train_time_steps,
        Transport=Transport,
        Stoichiometry=Stoichiometry,
        rxn_ids=rxn_ids,
        biomass_rxn_id=biomass_rxn_id,
        hidden_layers_lag=hidden_layers_lag,
        hidden_layers_flux=hidden_layers_flux,
        UB_in=UB_in, UB_out=UB_out,
        lag_function=lag_function,
        dropout_rate=dropout_rate,
        loss_weight = loss_weight,
        loss_decay = loss_decay,
        train_test_split=train_test_split,
        x_fold=x_fold,
        verbose=verbose,
        name='metabolic_model'
        )
        
        # Set concentration hold-out index (if any) based on train_test_split
        model.holdout_index = data.get_holdout_index_from_split(train_test_split, metabolite_ids)

        # Ensure the model is built before loading weights
        dummy_input = tf.constant(
            np.zeros((1, len(times), len(metabolite_ids)), dtype=np.float32)
        )
        model(dummy_input, training=False)
        model.load_weights(weights_path)

        if verbose:
            print(f'Model built from {config_path} and weights loaded from {weights_path}')
            
        return model

###############################################################################
# CREATE TRAIN AND CROSS-VALIDATE MODEL
###############################################################################

def reset_model(
    model,
    verbose=False
):
    """
    Reset provided model: returns a new model instance with same architecture and config.
    """

    # Use model attributes directly
    new_model = MetabolicModel(
        times=model.times,
        metabolite_ids=model.metabolite_ids,
        Transport=model.Transport,
        Stoichiometry=model.Stoichiometry,
        rxn_ids=model.rxn_ids,
        biomass_rxn_id=model.biomass_rxn_id,
        hidden_layers_lag=model.hidden_layers_lag,
        hidden_layers_flux=model.hidden_layers_flux,
        dropout_rate=model.dropout_rate,
        loss_weight=model.loss_weight,
        loss_decay=model.loss_decay,
        train_test_split=model.train_test_split,
        x_fold=model.x_fold,
        train_time_steps=model.train_time_steps,
        lag_function=getattr(model, "lag_function", 'exponential'), 
        UB_in=getattr(model, "UB_in", 0.0),
        UB_out=getattr(model, "UB_out", 0.0),
        verbose=verbose
    )

    # Force model to build
    dummy = np.zeros((1, len(model.times), len(model.metabolite_ids)), dtype=np.float32)
    _ = new_model(dummy, training=False)
    if verbose:
        new_model.printout()

    return new_model
    
def create_model_train_val(
    media_file, od_file,
    cobra_model_file,
    biomass_rxn_id,
    x_fold=5, 
    hidden_layers_lag=[0],
    hidden_layers_flux=[460], 
    dropout_rate=0.2,
    loss_weight=[0.001, 1, 1, 1],
    loss_decay=[0, 0.5, 0.5, 0.5],
    train_test_split='medium',
    UB_in=0, UB_out=0,
    lag_function='exponential',
    verbose=False
    ):
    """
    Prepares training and validation data and model for two split strategies:
      - 'medium': Split by random subsets of media/conditions (current default).
      - 'forecast': Temporal split within each medium, training on the first fraction
                    of time steps, validating (forecasting) on the last.

    Constraints
    ----------
    - 'medium' requires at least 2 experiments (Z >= 2).
    - 'forecast' requires at least 2 time points (T >= 2).

    Returns
    -------
    model, train_array, train_dev, val_array, val_dev, val_ids
        If 'forecast', val_ids will be time indices used for test set.
        If 'medium', val_ids are experiment IDs.
    """
    # Load and process data 
    times, metabolite_ids, experiment_data, dev_data, \
    Stoichiometry, Transport, rxn_ids = data.process_data(
        media_file, od_file, cobra_model_file, biomass_rxn_id, verbose=verbose
    )
    exp_ids = list(experiment_data.keys())
    Z = len(exp_ids)
    total_time_steps = len(times)

    # --- Sanity checks specific to split mode ---
    if train_test_split == 'medium' and Z < 2:
        raise ValueError(
            "train_test_split='medium' requires at least 2 experiments (Z >= 2). "
            "You currently have Z = 1. Use train_test_split='forecast' for single-experiment data "
            "(e.g. Millard)."
        )

    if train_test_split == 'forecast' and total_time_steps < 2:
        raise ValueError(
            "train_test_split='forecast' requires at least 2 time points (T >= 2). "
            f"You currently have T = {total_time_steps}."
        )

    # dev_data as array (may be empty dict, so guard)
    dev_values = list(dev_data.values())
    dev_data_arr = np.asarray(dev_values) if len(dev_values) > 0 else None
    
    base_split = train_test_split if train_test_split in ('medium', 'forecast') else 'forecast'

    if base_split == 'medium':  # Splitting by media
        np.random.shuffle(exp_ids)
        split = int(len(exp_ids) * (1 - 1/x_fold)) if x_fold > 1 else len(exp_ids)
        train_time_steps = total_time_steps
        train_ids = exp_ids[:split]
        train_data = {eid: experiment_data[eid] for eid in train_ids}
        train_array = data.prepare_experiment_array(total_time_steps, metabolite_ids, train_data)
        dev_data_arr = np.asarray(list(dev_data.values()))
        train_dev = dev_data_arr[:split]
        if x_fold > 1:
            val_ids = exp_ids[split:]
            val_data = {eid: experiment_data[eid] for eid in val_ids}
            val_array = data.prepare_experiment_array(total_time_steps, metabolite_ids, val_data)
            val_dev = dev_data_arr[split:]
        else:
            val_array, val_dev, val_ids = train_array, train_dev, train_ids
    elif base_split == 'forecast':
        # Time-based splitting (forecast-style). For concentration-holdout modes
        # (train_test_split starting with 'concentration-'), we override this
        # and use the FULL time-series for training (no temporal hold-out).
        if isinstance(train_test_split, str) and train_test_split.lower().startswith('concentration-'):
            # Concentration mode: no time-based split, train on all time points.
            train_time_steps = total_time_steps
            train_ids = exp_ids
            train_data = {eid: experiment_data[eid] for eid in train_ids}
            train_array = data.prepare_experiment_array(total_time_steps, metabolite_ids, train_data)

            # dev_data_arr already computed above (may be None)
            if dev_data_arr is not None:
                train_dev = dev_data_arr[:, :train_time_steps]
            else:
                train_dev = None
            # In concentration mode, validation is identical to training w.r.t. time axis.
            # (When there is only one experiment, this is effectively train==val.)
            val_array, val_dev, val_ids = train_array, train_dev, train_ids
        else:
            # Standard forecast mode: use the first fraction of time points for training,
            # and (optionally) all time points for validation.
            train_time_steps = int(round(total_time_steps * (x_fold - 1) / x_fold)) if x_fold > 1 else total_time_steps
            train_ids = exp_ids
            train_data = {eid: experiment_data[eid] for eid in train_ids}
            train_array = data.prepare_experiment_array(train_time_steps, metabolite_ids, train_data)
            dev_data_arr = np.asarray(list(dev_data.values()))
            train_dev = dev_data_arr[:, :train_time_steps]
            if x_fold > 1:
                val_ids = train_ids
                val_array = data.prepare_experiment_array(total_time_steps, metabolite_ids, train_data)
                val_dev = dev_data_arr
            else:
                val_array, val_dev, val_ids = train_array, train_dev, train_ids

    else:
        raise ValueError(f"Unknown base_split value: {base_split}")
        
    # Instantiate the model 
    model = MetabolicModel(
        times, 
        metabolite_ids,
        Transport, Stoichiometry, rxn_ids, biomass_rxn_id, 
        hidden_layers_lag=hidden_layers_lag, 
        hidden_layers_flux=hidden_layers_flux, 
        dropout_rate=dropout_rate, 
        loss_weight=loss_weight, 
        loss_decay=loss_decay,
        train_test_split=train_test_split, 
        x_fold=x_fold,
        train_time_steps=train_time_steps,
        UB_in=UB_in, UB_out=UB_out,
        lag_function=lag_function,
        verbose=verbose
    )
    # Set concentration hold-out index (if any) based on train_test_split
    model.holdout_index = data.get_holdout_index_from_split(train_test_split, metabolite_ids)
    
    if verbose:
        print(f'Train shape: {train_array.shape}')
        if val_array is not None:
            print(f'Val shape: {val_array.shape}')
        dummy = np.zeros((1, model.T, model.k), dtype=np.float32)
        _ = model(dummy, training=False)
        model.printout()
        
    return model, train_array, train_dev, val_array, val_dev, val_ids

def train_step(model, C_ref_batch, optimizer, verbose=False):
    with tf.GradientTape() as tape:
        loss_s_v, loss_neg_v, loss_c, loss_drop_c = model(C_ref_batch, training=True, verbose=verbose)
        loss = loss_s_v + loss_neg_v + loss_c + loss_drop_c  
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_s_v, loss_neg_v, loss_c, loss_drop_c

def train_model(
    model, 
    train_array, val_array=None,  # shape: (Z, (T+1)*k) or (Z, T_train*k) in 'forecast' mode
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    num_epochs=10, batch_size=10, patience=10,
    verbose=False,
    train_test_split='medium',
    x_fold=5
):
    """
    Trains the model, supporting 'medium' and 'forecast' split.
    """
    Z = train_array.shape[0]
    
    base_split = train_test_split if train_test_split in ('medium', 'forecast') else 'forecast'

    if base_split == 'medium':
        # Standard reshape as before
        C_train_all = train_array.reshape((Z, model.T, model.k))
        if val_array is not None:
            Z_val = val_array.shape[0]
            C_val_all = val_array.reshape((Z_val, model.T, model.k))
        else:
            C_val_all = None

    elif base_split == 'forecast':
        # Infer steps from shape and model.k
        T_train = train_array.shape[1] // model.k
        C_train_all = train_array.reshape((Z, T_train, model.k))
        if val_array is not None:
            Z_val = val_array.shape[0]
            T_val = val_array.shape[1] // model.k
            C_val_all = val_array.reshape((Z_val, T_val, model.k))
        else:
            C_val_all = None

    else:
        raise ValueError(f"Unknown base_split: {base_split}")


    # === SANITY CHECK: are we training on the right concentrations? ===
    if verbose:
        print("\n=== Sanity check: training data ===")
        print(f"C_train_all shape: {C_train_all.shape}  (Z, T_train, k)")
        print(f"Number of metabolites k = {model.k}")
        print(f"Number of time points in training trajectories = {C_train_all.shape[1]}")
        print(f"First 5 metabolite IDs (if available): {model.metabolite_ids[:3] if hasattr(model, 'metabolite_ids') else 'N/A'}")
        print(f"Biomass metabolite ID (last column): {model.metabolite_ids[-1] if hasattr(model, 'metabolite_ids') else 'N/A'}")

        ex0 = C_train_all[0]  # first experiment
        print('C_train_all.shape:', C_train_all.shape)
        T_show = min(20, ex0.shape[0])

        print("\nExperiment 0 — first time points:")
        for t_idx in range(T_show):
            t_val = float(model.times[t_idx]) if hasattr(model, "times") else t_idx
            row = ex0[t_idx]
            # show first few media components + biomass
            first_mets = row[:5]
            biomass = row[-1]
            od = utils.concentration_to_logOD(biomass)
            print(
                f"t = {t_val:6.2f}  "
                f"C[0,{t_idx},0:3] = {first_mets}  "
                f"BIOMASS = {biomass:.4f} "
                f"OD = {od:.4f}"
            )
            
        if C_val_all is not None:
            print("\n=== Sanity check: validation data ===")
            print(f"C_val_all shape: {C_val_all.shape}  (Z_val, T_val, k)")
        print("=== End sanity check ===\n")
    # === END SANITY CHECK ===

    # Track losses
    losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train = [], [], [], []
    losses_s_v_val, losses_neg_v_val, losses_c_val, losses_drop_c_val = [], [], [], []

    best_val_loss = np.inf
    best_weights = None
    steps_no_improv = 0
    n_batches = int(np.ceil(Z / batch_size))
    
    for epoch in range(num_epochs):
        idxs = np.arange(Z)
        np.random.shuffle(idxs)
        C_train_all = C_train_all[idxs]

        # Per-epoch sums
        epoch_loss_s_v, epoch_loss_neg_v, epoch_loss_c, epoch_loss_drop_c = 0.0, 0.0, 0.0, 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, Z)
            C_batch = C_train_all[start:end]
            # Pass the flag and x_fold to train_step
            loss_s_v, loss_neg_v, loss_c, loss_drop_c = train_step(
                model, C_batch, optimizer, verbose=verbose
            )
            epoch_loss_s_v += loss_s_v.numpy()
            epoch_loss_neg_v += loss_neg_v.numpy()
            epoch_loss_c += loss_c.numpy()
            epoch_loss_drop_c += loss_drop_c.numpy()

        # Average over batches
        epoch_loss_s_v /= n_batches
        epoch_loss_neg_v /= n_batches
        epoch_loss_c /= n_batches
        epoch_loss_drop_c /= n_batches

        losses_s_v_train.append(epoch_loss_s_v)
        losses_neg_v_train.append(epoch_loss_neg_v)
        losses_c_train.append(epoch_loss_c)
        losses_drop_c_train.append(epoch_loss_drop_c)

        if C_val_all is not None:
            val_s_v, val_neg_v, val_c, val_drop_c = model(C_val_all, training=False)
            val_s_v_np = val_s_v.numpy()
            val_neg_v_np = val_neg_v.numpy()
            val_c_np = val_c.numpy()
            val_drop_c_np = val_drop_c.numpy()
            losses_s_v_val.append(val_s_v_np)
            losses_neg_v_val.append(val_neg_v_np)
            losses_c_val.append(val_c_np)
            losses_drop_c_val.append(val_drop_c_np)

            val_loss = val_s_v_np + val_neg_v_np + val_c_np + val_drop_c_np
            if verbose:
                print(f'[Epoch {epoch+1}/{num_epochs}] '
                  f'Train: s_v={epoch_loss_s_v:.1e}, neg_v={epoch_loss_neg_v:.1e}, c={epoch_loss_c:.1e}, drop_c={epoch_loss_drop_c:.1e} | '
                  f'Val: s_v={val_s_v_np:.1e}, neg_v={val_neg_v_np:.1e}, c={val_c_np:.1e}, drop_c={val_drop_c_np:.1e}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights()
                steps_no_improv = 0
            else:
                steps_no_improv += 1
                if steps_no_improv >= patience:
                    if verbose:
                        print('Early stopping triggered. Restoring best weights.')
                    model.set_weights(best_weights)
                    break
        else:
            if verbose:
                print(f'[Epoch {epoch+1}/{num_epochs}] '
                  f'Train: s_v={epoch_loss_s_v:.1e}, neg_v={epoch_loss_neg_v:.1e}, c={epoch_loss_c:.1e}, drop_c={epoch_loss_drop_c:.1e}')

    return (
        (losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train),
        (losses_s_v_val, losses_neg_v_val, losses_c_val, losses_drop_c_val)
    )

###############################################################################
# PREDICT TIMECOURSE FOR VALIDATION 
###############################################################################

def predict_timecourse(model, C_ref_batch, verbose=False):
    """
    Predict the full timecourse of metabolite concentrations for a batch.

    Parameters
    ----------
    model : MetabolicModel
        Trained model instance.
    C_ref_batch : tf.Tensor or np.ndarray
        Reference concentrations of shape (batch_size, T, k),
        where T should match model.T and k = model.k.
        Works for both OD (k=1) and multi-metabolite (e.g. Millard, k=3).

    Returns
    -------
    tf.Tensor
        Predicted concentrations of shape (batch_size, T, k).
    """
    C_ref_batch = tf.convert_to_tensor(C_ref_batch, dtype=tf.float32)
    batch_size = tf.shape(C_ref_batch)[0]
    T = tf.shape(C_ref_batch)[1]

    C_pred = C_ref_batch[:, 0, :]   # (batch, k)
    lag_params = None

    predictions = tf.TensorArray(tf.float32, size=T)
    predictions = predictions.write(0, C_pred)

    for t in tf.range(T - 1):
        ttf = tf.cast(t, tf.float32)
        C_next, v_t, lag_params = model._next_concentration(
            C_pred, ttf, training=False, lag_params=lag_params, verbose=verbose
        )
        predictions = predictions.write(t + 1, C_next)
        C_pred = C_next

    predictions_stacked = predictions.stack()           # (T, batch, k)
    predictions_stacked = tf.transpose(predictions_stacked, [1, 0, 2])  # (batch, T, k)
    return predictions_stacked


def predict_on_val_data(
    model,
    val_array,
    verbose=False
):
    """
    Predict the timecourse for the validation data.

    Expects val_array to encode Z validation trajectories flattened along time:
      - shape (Z, T * k)  (multi-experiment case), or
      - shape (T * k,)    (single-experiment case, e.g. after np.loadtxt)

    Returns
    -------
    pred : tf.Tensor of shape (Z, T, k)
    ref  : np.ndarray of shape (Z, T, k)
    """

    arr = np.asarray(val_array)

    # Handle single-experiment serialization: (T*k,) -> (1, T*k)
    if arr.ndim == 1:
        arr = arr[None, :]

    Z_val, flat_dim = arr.shape
    k = model.k

    if flat_dim % k != 0:
        raise ValueError(
            f"Cannot reshape val_array of shape {arr.shape} "
            f"into (Z, T, k) with k={k}."
        )

    T_inferred = flat_dim // k

    # Optional consistency check against model.T
    if hasattr(model, "T") and T_inferred != model.T:
        # Not fatal, but worth warning
        print(
            f"[predict_on_val_data] Warning: inferred T={T_inferred} "
            f"differs from model.T={model.T}. Using T_inferred."
        )

    ref = arr.reshape((Z_val, T_inferred, k))
    pred = predict_timecourse(model, ref, verbose=verbose)
    return pred, ref


