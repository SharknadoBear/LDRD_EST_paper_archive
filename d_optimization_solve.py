"""
Simple WMCR leaching column solver (Cin = 0, constant Q = average Q).

States:
  Cw(t): mobile porewater / effluent concentration [ng/mL]
  Cs(t): solid concentration per solid volume        [ng/mL-solid]
  Mout(t): cumulative effluent mass                  [ng]

WMCR model:
  dCw/dt = -(Q/Vw)*Cw + (kA/Vw)*(Cs/Ksw - Cw)
  dCs/dt = -(kA/Vs)*(Cs/Ksw - Cw)
  dMout/dt = Q*Cw

Inputs (fixed):
  Q_avg_mL_min : average flow rate [mL/min]
  T_end_days   : experiment duration [days]
  Vw_mL        : porewater (mobile) volume [mL]
  m_s_g        : TWP mass [g]
  rho_s_g_cm3  : TWP density [g/cm^3]  (1 cm^3 = 1 mL)
  M_inf_ng     : total leachable mass (capacity) [ng]

Parameters:
  Ksw          : solid-water partition coefficient (Cs/Cw) [-]
  kA_mL_day    : exchange capacity (k * A_total) [mL/day]
"""

from dataclasses import dataclass
import os

# Keep each optimizer process from spawning many hidden BLAS/OpenMP threads.
# This matters on Windows because the optimizer can otherwise look unresponsive
# or hit paging-file limits before Python prints useful progress.
for _thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

if __name__ == "__main__":
    print("Importing scientific Python packages...", flush=True)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import json
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


N_OPT_CALLS = 50
N_INITIAL_POINTS = 10
PROGRESS_REPORT_INTERVAL = 10


def report_gp_progress(result):
    n_completed = len(result.func_vals)
    if (
        n_completed == 1
        or n_completed % PROGRESS_REPORT_INTERVAL == 0
        or n_completed == N_OPT_CALLS
    ):
        best_objective = np.min(result.func_vals)
        print(
            f"  gp_minimize progress: {n_completed}/{N_OPT_CALLS} calls, "
            f"best objective={best_objective:.6f}",
            flush=True,
        )


@dataclass
class WMCRInputs:
    Q_avg_mL_min: float
    T_end_days: float
    Vw_mL: float
    m_s_g: float
    rho_s_g_cm3: float
    M_inf_ng: float
    exp_tag: str = ""  # experimental data tag (e.g., "PPDQ_20C_M")


@dataclass
class WMCRParams:
    Ksw: float
    kA_mL_day: float


def solve_wmcr(inputs: WMCRInputs, params: WMCRParams, n_eval: int = 2001):
    # --- Basic checks
    if inputs.Q_avg_mL_min <= 0:
        raise ValueError("Q_avg_mL_min must be > 0")
    if inputs.T_end_days <= 0:
        raise ValueError("T_end_days must be > 0")
    if inputs.Vw_mL <= 0:
        raise ValueError("Vw_mL must be > 0")
    if inputs.m_s_g <= 0 or inputs.rho_s_g_cm3 <= 0:
        raise ValueError("m_s_g and rho_s_g_cm3 must be > 0")
    if inputs.M_inf_ng < 0:
        raise ValueError("M_inf_ng must be >= 0")
    if params.Ksw <= 0 or params.kA_mL_day <= 0:
        raise ValueError("Ksw and kA_mL_day must be > 0")

    # --- Derived quantities
    Q_day = inputs.Q_avg_mL_min * 1440.0  # mL/day
    Vs_mL = inputs.m_s_g / inputs.rho_s_g_cm3  # mL (since 1 cm^3 = 1 mL)

    # Initial conditions
    Cw0 = 0.0
    Cs0 = inputs.M_inf_ng / Vs_mL if Vs_mL > 0 else 0.0  # ng/mL-solid
    Mout0 = 0.0

    def rhs(t, y):
        Cw, Cs, Mout = y
        # driving force in water concentration units:
        #   (Cs/Ksw - Cw) has units ng/mL
        drive = (Cs / params.Ksw) - Cw

        dCw = -(Q_day / inputs.Vw_mL) * Cw + (params.kA_mL_day / inputs.Vw_mL) * drive
        dCs = -(params.kA_mL_day / Vs_mL) * drive
        dMout = Q_day * Cw
        return [dCw, dCs, dMout]

    t_eval = np.linspace(0.0, inputs.T_end_days, n_eval)
    y0 = [Cw0, Cs0, Mout0]

    sol = solve_ivp(
        rhs,
        t_span=(0.0, inputs.T_end_days),
        y0=y0,
        t_eval=t_eval,
        method="BDF",     # robust for stiff regimes (e.g., large kA)
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    Cw = sol.y[0]
    Cs = sol.y[1]
    Mout = sol.y[2]

    # Useful outputs
    Cout = Cw
    Ms = Cs * Vs_mL            # remaining mass in solid [ng]
    Mtotal = Ms + inputs.Vw_mL * Cw + Mout  # mass balance check (should be ~ M_inf_ng)

    return {
        "t_days": sol.t,
        "Cout_ng_mL": Cout,
        "Cs_ng_mL_solid": Cs,
        "Mout_ng": Mout,
        "Ms_ng": Ms,
        "mass_balance_ng": Mtotal,
        "Q_mL_day": Q_day,
        "Vs_mL": Vs_mL,
        "Cs0_ng_mL_solid": Cs0,
    }


def model_run():
    # --- Load Q data and calculate average
    data_dir = "./data"
    config_dir = "./config"
    exp_tag = "4C_S"  # experimental data tag (same for PPD and PPDQ)
    exp_full_tag = "PPDQ_4C_S"  # full tag for mass/concentration data files
    
    # Read Q data with MPT time ticks
    q_file = os.path.join(data_dir, f"Q_{exp_tag}.csv")
    q_df = pd.read_csv(q_file, header=None)
    q_values = q_df.iloc[:, 0].values  # First column in mL/min
    Q_avg_mL_min = np.mean(q_values)
    print(f"Calculated Q_avg from Q_{exp_tag}.csv: {Q_avg_mL_min:.4f} mL/min")
    
    # Read configuration from JSON file
    config_file = os.path.join(config_dir, f"meta_{exp_full_tag}.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract parameters for the specific tag (JSON uses "20C_M" as key regardless of filename)
    config_key = list(config.keys())[0]  # Get the first (and typically only) key
    params_dict = config[config_key]
    T_end_days = params_dict["T_end_days"]
    Vw_mL = params_dict["Vw_mL"]
    m_s_g = params_dict["m_s_g"]
    rho_s_g_cm3 = params_dict["rho_s_g_cm3"]
    M_inf_ng = params_dict["M_inf"]
    N_valid_start = params_dict.get("N_valid_start", 1)  # Default to 1 if not specified
    log_kA_low = params_dict["log_kA_low"]
    log_kA_high = params_dict["log_kA_high"]
    wall_delta_log10 = 0.025
    valid_log_kA_low = log_kA_low + wall_delta_log10
    valid_log_kA_high = log_kA_high - wall_delta_log10
    if valid_log_kA_low >= valid_log_kA_high:
        raise ValueError(
            f"Wall-hit exclusion delta is too large for log_kA bounds: "
            f"[{log_kA_low}, {log_kA_high}] with delta {wall_delta_log10}"
        )
    
    print(f"Loaded config from {config_file}")
    print(f"  T_end_days: {T_end_days}, Vw_mL: {Vw_mL}, m_s_g: {m_s_g}")
    print(f"  rho_s_g_cm3: {rho_s_g_cm3}, M_inf_ng: {M_inf_ng}")
    print(f"  N_valid_start: {N_valid_start}")
    print(f"  log_kA bounds: [{log_kA_low}, {log_kA_high}]")
    print(f"  Wall-hit exclusion delta: {wall_delta_log10} log10(kA) units")
    print(f"  Valid interior log_kA range: ({valid_log_kA_low}, {valid_log_kA_high})")
    
    # --- Create inputs
    inputs = WMCRInputs(
        Q_avg_mL_min=Q_avg_mL_min,
        T_end_days=T_end_days,
        Vw_mL=Vw_mL,
        m_s_g=m_s_g,
        rho_s_g_cm3=rho_s_g_cm3,
        M_inf_ng=M_inf_ng,
        exp_tag=exp_full_tag,
    )

    # --- Load experimental data
    exp_mass_time = None
    exp_mass = None
    exp_conc_time = None
    exp_conc = None
    
    if inputs.exp_tag:
        try:
            # Read time ticks
            ept_df = pd.read_csv(os.path.join(data_dir, "EPT.csv"), header=None)
            mpt_df = pd.read_csv(os.path.join(data_dir, "MPT.csv"), header=None)
            exp_mass_time_full = ept_df.iloc[:, 0].values  # First column in days
            exp_conc_time_full = mpt_df.iloc[:, 0].values  # First column in days
            
            # Read mass data (e.g., PPDQ_20C_M.csv)
            mass_file = os.path.join(data_dir, f"{inputs.exp_tag}.csv")
            mass_df = pd.read_csv(mass_file, header=None)
            exp_mass_full = mass_df.iloc[:, 0].values  # First column in ng
            
            # Read concentration data (e.g., C_PPDQ_20C_M.csv)
            conc_file = os.path.join(data_dir, f"C_{inputs.exp_tag}.csv")
            conc_df = pd.read_csv(conc_file, header=None)
            exp_conc_full = conc_df.iloc[:, 0].values  # First column in ng/L
            
            # Apply N_valid_start filter (convert from 1-based to 0-based index)
            start_idx = N_valid_start - 1  # Convert to 0-based index
            exp_mass_time = exp_mass_time_full[start_idx:]
            exp_mass = exp_mass_full[start_idx:]
            exp_conc_time = exp_conc_time_full[start_idx:]
            exp_conc = exp_conc_full[start_idx:]
            
            print(f"Loaded experimental data for tag: {inputs.exp_tag}")
            print(f"  Using data from index {N_valid_start} onwards (Python index {start_idx})")
            print(f"  Mass data points: {len(exp_mass)} (filtered from {len(exp_mass_full)})")
            print(f"  Concentration data points: {len(exp_conc)} (filtered from {len(exp_conc_full)})")
        except Exception as e:
            print(f"Warning: Could not load experimental data: {e}")
            return
    
    # --- Define objective function with adjustable weights
    def objective_function(log_Ksw, log_kA, w_mass=0.5, w_conc=0.5):
        """
        Objective function for optimization.
        Uses log-scale parameters and normalized RMSE.
        
        Parameters:
        -----------
        log_Ksw : float
            log10(Ksw)
        log_kA : float
            log10(kA_mL_day)
        w_mass : float
            Weight for mass RMSE (default 0.5)
        w_conc : float
            Weight for concentration RMSE (default 0.5)
        
        Returns:
        --------
        float : Combined normalized RMSE
        """
        try:
            Ksw = 10 ** log_Ksw
            kA_mL_day = 10 ** log_kA
            
            params = WMCRParams(Ksw=Ksw, kA_mL_day=kA_mL_day)
            out = solve_wmcr(inputs, params)
            
            # Interpolate WMCR outputs to experimental time points
            wmcr_mass_interp = interp1d(out["t_days"], out["Mout_ng"], 
                                        kind='linear', fill_value='extrapolate')
            wmcr_conc_interp = interp1d(out["t_days"], out["Cout_ng_mL"] * 1000,  # Convert to ng/L
                                        kind='linear', fill_value='extrapolate')
            
            wmcr_mass_at_exp = wmcr_mass_interp(exp_mass_time)
            wmcr_conc_at_exp = wmcr_conc_interp(exp_conc_time)
            
            # Calculate RMSE for mass and concentration
            rmse_mass = np.sqrt(np.mean((wmcr_mass_at_exp - exp_mass) ** 2))
            rmse_conc = np.sqrt(np.mean((wmcr_conc_at_exp - exp_conc) ** 2))
            
            # Normalize by mean of experimental data to make them comparable
            norm_rmse_mass = rmse_mass / np.mean(exp_mass) if np.mean(exp_mass) > 0 else rmse_mass
            norm_rmse_conc = rmse_conc / np.mean(exp_conc) if np.mean(exp_conc) > 0 else rmse_conc
            
            # Combined objective with weights
            objective = w_mass * norm_rmse_mass + w_conc * norm_rmse_conc
            
            return objective
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e10  # Return large value on error
    
    # --- Bayesian Optimization with Clustering
    print("\n" + "="*60)
    print("Starting Cluster Bayesian Optimization with Gaussian Process")
    print("="*60)
    
    # User input for number of valid cluster samples to collect
    N_cluster = int(input("Enter number of valid optimization clusters to collect (e.g., 5): "))
    
    # Define search space in log scale (around initial guesses)
    # Initial: Ksw = 0.32e4 = 3200, log10(3200) ≈ 3.51
    # Initial: kA_mL_day = 2e1 = 20, log10(20) ≈ 1.30
    
    space = [
        Real(-3.0, 6.0, name='log_Ksw'),      # 10^2.0 to 10^5.0 (1e-3 to 100000)
        Real(log_kA_low, log_kA_high, name='log_kA'),  # bounds from config: log10(kA_min) to log10(kA_max)
    ]
    
    # Wrapper for skopt
    @use_named_args(space)
    def objective_wrapper(**params):
        return objective_function(params['log_Ksw'], params['log_kA'], 
                                 w_mass=0.5, w_conc=0.5)  # Adjustable weights
    
    # Run optimization attempts until the requested number of valid clusters is collected.
    cluster_results = []
    cluster_outputs = []
    attempt_count = 0
    valid_count = 0
    wall_hit_count = 0
    
    print(f"\nCollecting {N_cluster} valid optimization clusters...")
    print(
        f"Wall-hit criterion: invalid if log_kA <= {valid_log_kA_low:.3f} "
        f"or log_kA >= {valid_log_kA_high:.3f}"
    )
    while valid_count < N_cluster:
        attempt_count += 1
        print(f"\n--- Optimization attempt {attempt_count} (valid {valid_count}/{N_cluster}) ---")
        
        # Run Bayesian optimization with different random state
        result = gp_minimize(
            objective_wrapper,
            space,
            n_calls=N_OPT_CALLS,  # Number of iterations
            n_initial_points=N_INITIAL_POINTS,  # Random exploration points
            acq_func="EI",  # Expected Improvement
            random_state=42 + attempt_count - 1,  # Different seed for each attempt
            verbose=False,
            callback=[report_gp_progress],
        )
        
        # Extract optimal parameters for this cluster
        log_Ksw = result.x[0]
        log_kA = result.x[1]
        Ksw = 10 ** log_Ksw
        kA = 10 ** log_kA
        low_wall_hit = log_kA <= valid_log_kA_low
        high_wall_hit = log_kA >= valid_log_kA_high
        is_valid = not (low_wall_hit or high_wall_hit)
        
        wall_status = "none"
        if low_wall_hit:
            wall_status = "lower kA wall"
        elif high_wall_hit:
            wall_status = "upper kA wall"
        print(
            f"Attempt {attempt_count} - Ksw: {Ksw:.2e}, kA: {kA:.2e}, "
            f"log_kA: {log_kA:.3f}, Objective: {result.fun:.6f}"
        )
        print(
            f"  Valid result: {is_valid} ({wall_status}); "
            f"valid count: {valid_count + (1 if is_valid else 0)}/{N_cluster}"
        )
        
        if not is_valid:
            wall_hit_count += 1
            continue
        
        valid_count += 1
        
        # Store only valid results
        cluster_results.append({
            'cluster_id': valid_count,
            'attempt_id': attempt_count,
            'Ksw': Ksw,
            'kA': kA,
            'log_Ksw': log_Ksw,
            'log_kA': log_kA,
            'objective': result.fun,
            'is_valid': True,
            'result': result
        })
        
        # Solve with this set of parameters
        params = WMCRParams(Ksw=Ksw, kA_mL_day=kA)
        out = solve_wmcr(inputs, params)
        cluster_outputs.append(out)
    
    print(f"\nCollected {valid_count} valid clusters after {attempt_count} attempts.")
    print(f"Boundary wall-hit attempts excluded: {wall_hit_count}")
    
    # Find the best cluster (minimum objective)
    best_idx = np.argmin([r['objective'] for r in cluster_results])
    best_result = cluster_results[best_idx]
    best_output = cluster_outputs[best_idx]
    
    print("\n" + "="*60)
    print("Cluster Optimization Results:")
    print("="*60)
    print(f"Best cluster: {best_result['cluster_id']}")
    print(f"Optimal Ksw: {best_result['Ksw']:.2e} (log10: {best_result['log_Ksw']:.3f})")
    print(f"Optimal kA_mL_day: {best_result['kA']:.2e} (log10: {best_result['log_kA']:.3f})")
    print(f"Minimum objective value: {best_result['objective']:.6f}")
    
    print(f"\nAll valid cluster results:")
    for res in cluster_results:
        print(
            f"  Cluster {res['cluster_id']} (attempt {res['attempt_id']}): "
            f"Ksw={res['Ksw']:.2e}, kA={res['kA']:.2e}, Obj={res['objective']:.6f}"
        )
    
    # --- Calculate uncertainty from all clusters
    t = best_output["t_days"]
    n_time = len(t)
    
    # Collect all WMCR predictions at the same time points
    all_mass = np.zeros((N_cluster, n_time))
    all_conc = np.zeros((N_cluster, n_time))
    
    for i, out in enumerate(cluster_outputs):
        # Interpolate to common time grid (from best solution)
        mass_interp = interp1d(out["t_days"], out["Mout_ng"], 
                               kind='linear', fill_value='extrapolate')
        conc_interp = interp1d(out["t_days"], out["Cout_ng_mL"] * 1000,  # Convert to ng/L
                               kind='linear', fill_value='extrapolate')
        
        all_mass[i, :] = mass_interp(t)
        all_conc[i, :] = conc_interp(t)
    
    # Calculate mean and std across clusters
    mass_mean = np.mean(all_mass, axis=0)
    mass_std = np.std(all_mass, axis=0)
    conc_mean = np.mean(all_conc, axis=0)
    conc_std = np.std(all_conc, axis=0)
    
    # Use best solution for plotting
    mass_best = best_output["Mout_ng"]
    conc_best = best_output["Cout_ng_mL"] * 1000  # Convert to ng/L

    # --- Plot optimized results with uncertainty shading
    fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Mass plot
    ax[0].plot(t, mass_best, label=f"Best WMCR (Ksw={best_result['Ksw']:.2e}, kA={best_result['kA']:.2e})", 
               linewidth=2, color='blue', zorder=3)
    ax[0].fill_between(t, mass_mean - mass_std, mass_mean + mass_std, 
                       alpha=0.3, color='blue', label=f'±1σ uncertainty (N={N_cluster})', zorder=1)
    if exp_mass is not None and exp_mass_time is not None:
        ax[0].plot(exp_mass_time, exp_mass, 'o', label="Experimental data", 
                   markersize=6, color='red', markerfacecolor='none', markeredgewidth=1.5, zorder=2)
    ax[0].set_ylabel("Cumulative mass in effluent [ng]", fontsize=11)
    ax[0].legend(fontsize=9, loc='best')
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title(f"Cluster Bayesian Optimization Results (N_valid={N_cluster}, attempts={attempt_count})", 
                   fontsize=12, fontweight='bold')

    # Concentration plot
    ax[1].plot(t, conc_best, label=f"Best WMCR", 
               linewidth=2, color='blue', zorder=3)
    ax[1].fill_between(t, conc_mean - conc_std, conc_mean + conc_std, 
                       alpha=0.3, color='blue', label=f'±1σ uncertainty (N={N_cluster})', zorder=1)
    if exp_conc is not None and exp_conc_time is not None:
        ax[1].plot(exp_conc_time, exp_conc, 'o', label="Experimental data", 
                   markersize=6, color='red', markerfacecolor='none', markeredgewidth=1.5, zorder=2)
    ax[1].set_xlabel("Time [days]", fontsize=11)
    ax[1].set_ylabel("Effluent concentration [ng/L]", fontsize=11)
    ax[1].legend(fontsize=9, loc='best')
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Create outputs directory if it doesn't exist
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main figure with experiment tag
    fig_file = os.path.join(output_dir, f"optimization_result_{exp_full_tag}.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: {fig_file}")
    plt.show()
    
    # --- Create boxplot for parameter distribution
    fig_box, ax_box = plt.subplots(1, 2, figsize=(12, 6))
    
    # Extract all Ksw and kA values from clusters
    all_Ksw = [r['Ksw'] for r in cluster_results]
    all_kA = [r['kA'] for r in cluster_results]
    
    # Boxplot for Ksw
    bp1 = ax_box[0].boxplot([all_Ksw], labels=['Ksw'], patch_artist=True, widths=0.5)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][0].set_edgecolor('blue')
    bp1['boxes'][0].set_linewidth(2)
    
    # Add optimal value marker and label
    ax_box[0].plot(1, best_result['Ksw'], 'r*', markersize=15, label='Optimal', zorder=3)
    ax_box[0].text(1.15, best_result['Ksw'], f"{best_result['Ksw']:.2e}", 
                   fontsize=10, color='red', fontweight='bold', va='center')
    ax_box[0].set_ylabel('Ksw [-]', fontsize=12, fontweight='bold')
    ax_box[0].set_title('Solid-Water Partition Coefficient', fontsize=11, fontweight='bold')
    ax_box[0].grid(True, alpha=0.3, axis='y')
    ax_box[0].legend(fontsize=10)
    
    # Add statistics text
    ksw_mean = np.mean(all_Ksw)
    ksw_std = np.std(all_Ksw)
    ax_box[0].text(0.98, 0.02, f'Mean: {ksw_mean:.2e}\nStd: {ksw_std:.2e}\nCV: {ksw_std/ksw_mean*100:.1f}%',
                   transform=ax_box[0].transAxes, fontsize=9, 
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Boxplot for kA
    bp2 = ax_box[1].boxplot([all_kA], labels=['kA'], patch_artist=True, widths=0.5)
    bp2['boxes'][0].set_facecolor('lightgreen')
    bp2['boxes'][0].set_edgecolor('green')
    bp2['boxes'][0].set_linewidth(2)
    
    # Add optimal value marker and label
    ax_box[1].plot(1, best_result['kA'], 'r*', markersize=15, label='Optimal', zorder=3)
    ax_box[1].text(1.15, best_result['kA'], f"{best_result['kA']:.2e}", 
                   fontsize=10, color='red', fontweight='bold', va='center')
    ax_box[1].set_ylabel('kA [mL/day]', fontsize=12, fontweight='bold')
    ax_box[1].set_title('Exchange Capacity', fontsize=11, fontweight='bold')
    ax_box[1].grid(True, alpha=0.3, axis='y')
    ax_box[1].legend(fontsize=10)
    
    # Add statistics text
    ka_mean = np.mean(all_kA)
    ka_std = np.std(all_kA)
    ax_box[1].text(0.98, 0.02, f'Mean: {ka_mean:.2e}\nStd: {ka_std:.2e}\nCV: {ka_std/ka_mean*100:.1f}%',
                   transform=ax_box[1].transAxes, fontsize=9, 
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig_box.suptitle(f'Parameter Distribution from {N_cluster} Valid Optimization Clusters\nExperiment: {exp_full_tag}',
                     fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # Save boxplot
    boxplot_file = os.path.join(output_dir, f"parameter_boxplot_{exp_full_tag}.png")
    plt.savefig(boxplot_file, dpi=150, bbox_inches='tight')
    print(f"Boxplot saved as: {boxplot_file}")
    plt.show()

    # --- Quick diagnostics
    mb_err = best_output["mass_balance_ng"] - inputs.M_inf_ng
    print(f"\nWMCR diagnostics:")
    print(f"Vs = {best_output['Vs_mL']:.4f} mL")
    print(f"Q  = {best_output['Q_mL_day']:.2f} mL/day")
    print(f"Max mass-balance error: {np.max(np.abs(mb_err)):.3e} ng")
    
    # Calculate final RMSE for best WMCR fit
    wmcr_mass_interp = interp1d(best_output["t_days"], best_output["Mout_ng"], 
                                kind='linear', fill_value='extrapolate')
    wmcr_conc_interp = interp1d(best_output["t_days"], best_output["Cout_ng_mL"] * 1000,
                                kind='linear', fill_value='extrapolate')
    
    wmcr_mass_at_exp = wmcr_mass_interp(exp_mass_time)
    wmcr_conc_at_exp = wmcr_conc_interp(exp_conc_time)
    
    rmse_mass = np.sqrt(np.mean((wmcr_mass_at_exp - exp_mass) ** 2))
    rmse_conc = np.sqrt(np.mean((wmcr_conc_at_exp - exp_conc) ** 2))
    
    print(f"\nFinal RMSE (Best WMCR):")
    print(f"Mass RMSE: {rmse_mass:.2f} ng")
    print(f"Concentration RMSE: {rmse_conc:.2f} ng/L")
    
    # Save cluster results to JSON with additional statistics
    output_results = {
        'experiment_tag': exp_full_tag,
        'experiment_conditions': {
            'Q_avg_mL_min': Q_avg_mL_min,
            'T_end_days': T_end_days,
            'Vw_mL': Vw_mL,
            'm_s_g': m_s_g,
            'rho_s_g_cm3': rho_s_g_cm3,
            'M_inf_ng': M_inf_ng,
            'N_valid_start': N_valid_start
        },
        'optimization_settings': {
            'N_clusters': N_cluster,
            'N_valid_clusters': valid_count,
            'N_attempts': attempt_count,
            'N_wall_hit_attempts_excluded': wall_hit_count,
            'wall_delta_log10': wall_delta_log10,
            'valid_log_kA_low': valid_log_kA_low,
            'valid_log_kA_high': valid_log_kA_high,
            'n_calls_per_cluster': N_OPT_CALLS,
            'n_initial_points': N_INITIAL_POINTS,
            'weight_mass': 0.5,
            'weight_conc': 0.5
        },
        'best_cluster': best_result['cluster_id'],
        'best_parameters': {
            'Ksw': best_result['Ksw'],
            'kA_mL_day': best_result['kA'],
            'log_Ksw': best_result['log_Ksw'],
            'log_kA': best_result['log_kA']
        },
        'best_objective': best_result['objective'],
        'parameter_statistics': {
            'Ksw': {
                'mean': float(np.mean(all_Ksw)),
                'std': float(np.std(all_Ksw)),
                'min': float(np.min(all_Ksw)),
                'max': float(np.max(all_Ksw)),
                'cv_percent': float(np.std(all_Ksw) / np.mean(all_Ksw) * 100)
            },
            'kA_mL_day': {
                'mean': float(np.mean(all_kA)),
                'std': float(np.std(all_kA)),
                'min': float(np.min(all_kA)),
                'max': float(np.max(all_kA)),
                'cv_percent': float(np.std(all_kA) / np.mean(all_kA) * 100)
            }
        },
        'all_clusters': [
            {
                'cluster_id': r['cluster_id'],
                'attempt_id': r['attempt_id'],
                'is_valid': r['is_valid'],
                'Ksw': r['Ksw'],
                'kA': r['kA'],
                'log_Ksw': r['log_Ksw'],
                'log_kA': r['log_kA'],
                'objective': r['objective']
            }
            for r in cluster_results
        ],
        'rmse': {
            'mass': float(rmse_mass),
            'concentration': float(rmse_conc)
        },
        'uncertainty': {
            'mass_std_mean': float(np.mean(mass_std)),
            'conc_std_mean': float(np.mean(conc_std))
        }
    }
    
    output_file = os.path.join(output_dir, f"optimization_specs_{exp_full_tag}.json")
    with open(output_file, 'w') as f:
        json.dump(output_results, f, indent=2)
    print(f"\nOptimization specifications saved to: {output_file}")
    print(f"\nAll outputs saved to: {output_dir}/")
    print(f"  - Main figure: optimization_result_{exp_full_tag}.png")
    print(f"  - Boxplot: parameter_boxplot_{exp_full_tag}.png")
    print(f"  - Specifications: optimization_specs_{exp_full_tag}.json")


if __name__ == "__main__":
    model_run()
