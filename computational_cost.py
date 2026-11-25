import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
BASE_DIR_OUTPUT = r"C:\Users\Victor Higino\Documents\Doutorado\CCAE"
RESULTS_SAVE_DIR = os.path.join(BASE_DIR_OUTPUT, "computational_efficiency_results")

N_FOLDS = 4

# Fallback max epochs per fold if exact early-stopped epoch counts are not stored in SQLite
DEFAULT_EPOCHS_PER_FOLD = {
    "Yellow Frame": 200,
    "Z24 Bridge": 200,
}
FORCE_EPOCHS_PER_FOLD = {
    "Yellow Frame": None,  # Set an integer to override auto-detection
    "Z24 Bridge": None,
}

# Single selected regularization weight per model family for the main table row
SELECTED_WEIGHTS = {
    "Yellow Frame": {
        "CCAE": "1e-01",
        "Traditional CAE": "N/A",
        "VAE": "1e-01",
        "SAE-L1": "1e-03",
        "SAE-KL": "1e-03",
    },
    "Z24 Bridge": {
        "CCAE": "1e-01",
        "Traditional CAE": "N/A",
        "VAE": "1e-02",
        "SAE-L1": "1e-03",
        "SAE-KL": "1e-03",
    },
}

NUM_SENSORS_BY_DATASET = {
    "Yellow Frame": 15,
    "Z24 Bridge": 7,
}

DATASET_MODEL_CONFIGS = {
    "Yellow Frame": {
        "CCAE": {
            "type": "ccae",
            "multi_weight": True,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_yellow_frame"),
            ],
        },
        "Traditional CAE": {
            "type": "cae",
            "multi_weight": False,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_yellow_frame_CAE"),
            ],
        },
        "VAE": {
            "type": "vae",
            "multi_weight": True,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_yellow_frame_VAE"),
            ],
        },
        "SAE-L1": {
            "type": "sae_l1",
            "multi_weight": True,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_yellow_frame_SAE_L1"),
            ],
        },
        "SAE-KL": {
            "type": "sae_kl",
            "multi_weight": True,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_yellow_frame_SAE_kl"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_yellow_frame_SAE_KL"),
            ],
        },
    },
    "Z24 Bridge": {
        "CCAE": {
            "type": "ccae",
            "multi_weight": True,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_Z24"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_bridge"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_Z24_bridge"),
            ],
        },
        "Traditional CAE": {
            "type": "cae",
            "multi_weight": False,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_CAE"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_Z24_CAE"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_bridge_CAE"),
            ],
        },
        "VAE": {
            "type": "vae",
            "multi_weight": True,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_VAE"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_Z24_VAE"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_bridge_VAE"),
            ],
        },
        "SAE-L1": {
            "type": "sae_l1",
            "multi_weight": True,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_SAE_L1"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_Z24_SAE_L1"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_bridge_SAE_L1"),
            ],
        },
        "SAE-KL": {
            "type": "sae_kl",
            "multi_weight": True,
            "dir_candidates": [
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_SAE_kl"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_SAE_KL"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_Z24_SAE_kl"),
                os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_Z24_SAE_KL"),
            ],
        },
    },
}


# ==============================================================================
# 2. EXACT TRAINABLE PARAMETER COUNT
# ==============================================================================
def compute_trainable_params(model_type, num_sensors, bottleneck_channels=4, classifier_linear_dim=64):
    """Computes exact trainable parameter count matching each architecture in PyTorch."""
    if not TORCH_AVAILABLE:
        return np.nan

    def shared_encoder():
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
        )

    def shared_decoder(in_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid(),
        )

    modules = [shared_encoder()]
    if model_type == "ccae":
        modules.append(nn.Sequential(
            nn.Conv2d(128, bottleneck_channels, 3, 1, 1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
        ))
        modules.append(nn.Embedding(num_sensors, 64))
        modules.append(nn.Sequential(
            nn.Linear(bottleneck_channels * 8 * 8, classifier_linear_dim),
            nn.ReLU(),
            nn.Linear(classifier_linear_dim, num_sensors),
        ))
        modules.append(shared_decoder(bottleneck_channels + 1))
    elif model_type == "cae":
        modules.append(nn.Sequential(
            nn.Conv2d(128, bottleneck_channels, 3, 1, 1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
        ))
        modules.append(shared_decoder(bottleneck_channels))
    elif model_type == "vae":
        modules.append(nn.Conv2d(128, bottleneck_channels, 3, 1, 1))
        modules.append(nn.Conv2d(128, bottleneck_channels, 3, 1, 1))
        modules.append(shared_decoder(bottleneck_channels))
    elif model_type in ("sae_l1", "sae_kl"):
        modules.append(nn.Conv2d(128, bottleneck_channels, 3, 1, 1))
        modules.append(shared_decoder(bottleneck_channels))
    else:
        return np.nan

    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)


# ==============================================================================
# 3. OPTUNA STUDY DISCOVERY & TIMING / PRUNING / BATCH SIZE METRICS
# ==============================================================================
def resolve_existing_dir(candidates):
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def parse_weight_value(w_tag):
    try:
        return float(w_tag)
    except Exception:
        return float("inf")


def discover_all_studies(cfg):
    """Finds all SQLite (.db) studies for a model family (across all weight_* folders or root)."""
    base_dir = resolve_existing_dir(cfg["dir_candidates"])
    if base_dir is None:
        return []

    studies = []
    if cfg.get("multi_weight", False):
        weight_dirs = sorted(
            [d for d in glob.glob(os.path.join(base_dir, "weight_*")) if os.path.isdir(d)],
            key=lambda p: parse_weight_value(os.path.basename(p).replace("weight_", "")),
        )
        for sub_dir in weight_dirs:
            w_tag = os.path.basename(sub_dir).replace("weight_", "")
            db_files = sorted(glob.glob(os.path.join(sub_dir, "*.db")))
            if db_files:
                studies.append({"weight_label": w_tag, "db_path": db_files[0]})

    if not studies:
        root_dbs = sorted(glob.glob(os.path.join(base_dir, "*.db")))
        if root_dbs:
            studies.append({"weight_label": "N/A", "db_path": root_dbs[0]})

    return studies


def select_target_study(all_studies, target_weight):
    """Selects the single configured weight study for the main table row."""
    if not all_studies:
        return None
    for st in all_studies:
        if st["weight_label"] == target_weight:
            return st
    if len(all_studies) == 1:
        chosen = dict(all_studies[0])
        if target_weight not in ("", None):
            chosen["weight_label"] = target_weight
        return chosen
    return all_studies[0]


def get_trial_duration_sec(trial):
    if trial.duration is not None:
        return float(trial.duration.total_seconds())
    if trial.datetime_start is not None and trial.datetime_complete is not None:
        return float((trial.datetime_complete - trial.datetime_start).total_seconds())
    return np.nan


def infer_epochs_for_completed_trial(trial, dataset_name, n_folds=N_FOLDS):
    """Infers total epochs executed across all folds in a completed trial."""
    forced = FORCE_EPOCHS_PER_FOLD.get(dataset_name, None)
    if forced is not None:
        return float(forced * n_folds), f"Forced ({forced}/fold)"

    attrs = trial.user_attrs or {}
    for key in ("fold_epochs", "epochs_per_fold_list", "trained_epochs_per_fold"):
        if key in attrs and isinstance(attrs[key], (list, tuple)) and len(attrs[key]) > 0:
            return float(np.sum(attrs[key])), f"Logged (user_attrs['{key}'])"

    for key in ("total_epochs", "total_trained_epochs"):
        if key in attrs and attrs[key]:
            return float(attrs[key]), f"Logged (user_attrs['{key}'])"

    for key in ("epochs_per_fold", "num_epochs", "max_epochs"):
        if key in attrs and attrs[key]:
            return float(attrs[key]) * n_folds, f"Logged (user_attrs['{key}'])"

    params = trial.params or {}
    for key in ("num_epochs", "epochs", "n_epochs"):
        if key in params and params[key]:
            return float(params[key]) * n_folds, f"Logged (params['{key}'])"

    default_ep = DEFAULT_EPOCHS_PER_FOLD.get(dataset_name, 200)
    return float(default_ep * n_folds), f"Configured max ({default_ep}/fold)"


def calc_mean_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return m, s


def fmt_mean_std(values, decimals=2):
    m, s = calc_mean_std(values)
    if np.isnan(m):
        return "N/A", np.nan
    return f"{m:.{decimals}f} ± {s:.{decimals}f}", m


def get_pruned_stats_from_db(db_path):
    """Returns (n_pruned, n_total, pct_pruned) from a single Optuna SQLite database."""
    storage_url = f"sqlite:///{db_path}"
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    if not summaries:
        return np.nan, 0, np.nan
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage_url)
    all_trials = study.trials
    n_total = len(all_trials)
    n_pruned = sum(1 for t in all_trials if t.state == TrialState.PRUNED)
    pct_pruned = (100.0 * n_pruned / n_total) if n_total > 0 else np.nan
    return n_pruned, n_total, pct_pruned


def compute_cross_weight_pruned_summary(all_studies):
    """Computes Mean ± Std of pruned trials across all weight studies of a model family."""
    counts = []
    pcts = []
    for st in all_studies:
        n_pruned, n_total, pct = get_pruned_stats_from_db(st["db_path"])
        if n_total > 0 and not np.isnan(n_pruned):
            counts.append(float(n_pruned))
            pcts.append(float(pct))

    if len(counts) <= 1:
        return "--", (counts[0] if counts else np.nan), (pcts[0] if pcts else np.nan), len(counts)

    m_n, s_n = calc_mean_std(counts)
    m_p, _ = calc_mean_std(pcts)
    return f"{m_n:.1f} ± {s_n:.1f} ({m_p:.1f}%)", m_n, m_p, len(counts)


def analyze_study_efficiency(dataset_name, model_label, model_type, target_study, all_studies, n_folds=N_FOLDS):
    """Computes computational efficiency metrics for the selected weight study plus cross-weight mean pruned trials."""
    db_path = target_study["db_path"]
    storage_url = f"sqlite:///{db_path}"
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    if not summaries:
        return None

    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage_url)
    all_trials = study.trials

    completed = [t for t in all_trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in all_trials if t.state == TrialState.PRUNED]

    n_total = len(all_trials)
    n_pruned = len(pruned)
    pct_pruned = (100.0 * n_pruned / n_total) if n_total > 0 else 0.0

    comp_trial_sec = []
    fold_sec_list = []
    epoch_sec_list = []
    batch_sizes_comp = []
    epoch_sources = set()

    for t in completed:
        params = t.params or {}
        if "batch_size" in params and params["batch_size"] is not None:
            batch_sizes_comp.append(float(params["batch_size"]))

        dur_sec = get_trial_duration_sec(t)
        if np.isnan(dur_sec) or dur_sec <= 0:
            continue
        comp_trial_sec.append(dur_sec)

        attrs = t.user_attrs or {}
        if "fold_durations_sec" in attrs and attrs["fold_durations_sec"]:
            fold_sec_list.extend([float(x) for x in attrs["fold_durations_sec"]])
        elif "fold_times" in attrs and attrs["fold_times"]:
            fold_sec_list.extend([float(x) for x in attrs["fold_times"]])
        else:
            fold_sec_list.append(dur_sec / n_folds)

        tot_epochs, src_label = infer_epochs_for_completed_trial(t, dataset_name, n_folds=n_folds)
        epoch_sources.add(src_label)
        if tot_epochs > 0:
            epoch_sec_list.append(dur_sec / tot_epochs)

    pruned_trial_sec = [
        dur for t in pruned
        for dur in [get_trial_duration_sec(t)]
        if not np.isnan(dur) and dur > 0
    ]

    all_valid_sec = [
        dur for t in all_trials
        for dur in [get_trial_duration_sec(t)]
        if not np.isnan(dur) and dur > 0
    ]
    total_hpo_hours = float(np.sum(all_valid_sec) / 3600.0) if all_valid_sec else np.nan

    str_bs, mean_bs = fmt_mean_std(batch_sizes_comp, decimals=1)
    str_epoch, mean_epoch = fmt_mean_std(epoch_sec_list, decimals=2)
    str_fold, mean_fold = fmt_mean_std(np.array(fold_sec_list) / 60.0, decimals=2)
    str_comp, mean_comp = fmt_mean_std(np.array(comp_trial_sec) / 60.0, decimals=2)
    str_pruned_t, mean_pruned_t = fmt_mean_std(np.array(pruned_trial_sec) / 60.0, decimals=2)

    str_mean_pruned_all_w, _, _, n_weight_studies = compute_cross_weight_pruned_summary(all_studies)

    n_params = compute_trainable_params(model_type, NUM_SENSORS_BY_DATASET[dataset_name])
    params_k = f"{n_params / 1e3:.1f}" if not np.isnan(n_params) else "N/A"

    return {
        "Dataset": dataset_name,
        "Model": model_label,
        "Weight": target_study["weight_label"],
        "Params (x10^3)": params_k,
        "Avg. Batch Size": str_bs,
        "Time/Epoch (s)": str_epoch,
        "Time/Fold (min)": str_fold,
        "Time/Completed Trial (min)": str_comp,
        "Time/Pruned Trial (min)": str_pruned_t,
        "Pruned Trials N (%)": f"{n_pruned} ({pct_pruned:.1f}%)",
        "Mean Pruned Trials (All w)": str_mean_pruned_all_w,
        "Total HPO (h)": f"{total_hpo_hours:.2f}" if not np.isnan(total_hpo_hours) else "N/A",
        "_is_summary": False,
        "_num_bs": mean_bs,
        "_num_epoch_s": mean_epoch,
        "_num_fold_min": mean_fold,
        "_num_comp_min": mean_comp,
        "_num_pruned_min": mean_pruned_t,
        "_num_pruned_n": float(n_pruned),
        "_num_pruned_pct": float(pct_pruned),
        "_num_hpo_h": total_hpo_hours,
        "_n_weight_studies": n_weight_studies,
        "_epoch_source": ", ".join(sorted(epoch_sources)) if epoch_sources else "N/A",
    }


def build_dataset_mean_row(dataset_name, dataset_rows):
    """Builds a Dataset Mean ± Std summary row across all evaluated architectures."""
    str_bs, _ = fmt_mean_std([r["_num_bs"] for r in dataset_rows], decimals=1)
    str_epoch, _ = fmt_mean_std([r["_num_epoch_s"] for r in dataset_rows], decimals=2)
    str_fold, _ = fmt_mean_std([r["_num_fold_min"] for r in dataset_rows], decimals=2)
    str_comp, _ = fmt_mean_std([r["_num_comp_min"] for r in dataset_rows], decimals=2)
    str_pruned_t, _ = fmt_mean_std([r["_num_pruned_min"] for r in dataset_rows], decimals=2)
    str_hpo, _ = fmt_mean_std([r["_num_hpo_h"] for r in dataset_rows], decimals=2)

    m_pn, s_pn = calc_mean_std([r["_num_pruned_n"] for r in dataset_rows])
    m_pp, _ = calc_mean_std([r["_num_pruned_pct"] for r in dataset_rows])
    str_pruned_n = f"{m_pn:.1f} ± {s_pn:.1f} ({m_pp:.1f}%)" if not np.isnan(m_pn) else "N/A"

    return {
        "Dataset": dataset_name,
        "Model": "Dataset Mean",
        "Weight": "--",
        "Params (x10^3)": "--",
        "Avg. Batch Size": str_bs,
        "Time/Epoch (s)": str_epoch,
        "Time/Fold (min)": str_fold,
        "Time/Completed Trial (min)": str_comp,
        "Time/Pruned Trial (min)": str_pruned_t,
        "Pruned Trials N (%)": str_pruned_n,
        "Mean Pruned Trials (All w)": "--",
        "Total HPO (h)": str_hpo,
        "_is_summary": True,
    }


# ==============================================================================
# 4. LATEX EXPORT (ENGLISH, SINGLE UNIFIED TABLE)
# ==============================================================================
def format_weight_latex(w_str):
    if w_str in ("N/A", "--", "", None):
        return "--"
    m = re.match(r"^([0-9.]+)e([+-]?[0-9]+)$", str(w_str).strip(), re.IGNORECASE)
    if m:
        base, exp = float(m.group(1)), int(m.group(2))
        if np.isclose(base, 1.0):
            return f"$10^{{{exp}}}$"
        return f"${base:g}\\times 10^{{{exp}}}$"
    return str(w_str)


def escape_latex(val):
    return str(val).replace("±", r" $\pm$ ").replace("%", r"\%").replace("_", r"\_")


def export_latex_table(df_full, tex_path):
    """Exports a single publication-ready LaTeX table in English."""
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{Computational efficiency analysis of the hyperparameter optimization (HPO) process across all evaluated architectures on the Yellow Frame and Z24 Bridge benchmarks (4-fold cross-validation, 200 Optuna trials per study with median pruning).}",
        r"\label{tab:computational_efficiency}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Model ($w$)} & \textbf{Params} & \textbf{Avg. Batch} & \textbf{Time/Epoch} & \textbf{Time/Fold} & \textbf{Time/Comp. Trial} & \textbf{Time/Pruned Trial} & \textbf{Pruned Trials} & \textbf{Mean Pruned (All $w$)} & \textbf{Total HPO} \\",
        r" &  & \textbf{($\times 10^3$)} & \textbf{Size} & \textbf{(s)} & \textbf{(min)} & \textbf{(min)} & \textbf{(min)} & \textbf{$N$ (\%)} & \textbf{Mean $\pm$ Std (\%)} & \textbf{(h)} \\",
        r"\midrule",
    ]

    datasets = list(df_full["Dataset"].unique())
    for ds_idx, dataset_name in enumerate(datasets):
        grp = df_full[df_full["Dataset"] == dataset_name]
        n_rows = len(grp)
        for row_i, (_, r) in enumerate(grp.iterrows()):
            if r["_is_summary"]:
                lines.append(r"\cmidrule(lr){2-11}")
                ds_cell = ""
                model_disp = r"\textit{Dataset Mean}"
            else:
                ds_cell = rf"\multirow{{{n_rows}}}{{*}}{{\textbf{{{dataset_name}}}}}" if row_i == 0 else ""
                w_tex = format_weight_latex(r["Weight"])
                model_disp = f"{r['Model']} ({w_tex})"
                if r["Model"] == "CCAE":
                    model_disp = rf"\textbf{{{model_disp}}}"

            row_str = (
                f"{ds_cell} & {model_disp} & {r['Params (x10^3)']} & "
                f"{escape_latex(r['Avg. Batch Size'])} & "
                f"{escape_latex(r['Time/Epoch (s)'])} & "
                f"{escape_latex(r['Time/Fold (min)'])} & "
                f"{escape_latex(r['Time/Completed Trial (min)'])} & "
                f"{escape_latex(r['Time/Pruned Trial (min)'])} & "
                f"{escape_latex(r['Pruned Trials N (%)'])} & "
                f"{escape_latex(r['Mean Pruned Trials (All w)'])} & "
                f"{escape_latex(r['Total HPO (h)'])} \\\\"
            )
            lines.append(row_str)

        if ds_idx < len(datasets) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} All timing and batch-size metrics are reported as mean $\pm$ standard deviation. Avg. Batch Size (sampled from $\{32, 64, 128\}$), Time/Epoch, Time/Fold, and Time/Completed Trial are computed exclusively over completed Optuna trials (\texttt{COMPLETE}), whereas Time/Pruned Trial is computed over early-terminated trials (\texttt{PRUNED}). Pruned Trials reports the pruned count and percentage for the selected regularization weight $w$; Mean Pruned (All $w$) reports the mean $\pm$ standard deviation of pruned trials across all evaluated regularization weights of each model family (Traditional CAE has no regularization penalty, $n = 1$ study). Dataset Mean rows summarize the mean $\pm$ standard deviation across all five architectures per benchmark. All experiments were executed on an Intel Core Ultra 9 285K CPU, 64 GB RAM, and NVIDIA GeForce RTX 5070 GPU (12 GB VRAM).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    all_rows = []

    print("=" * 160)
    print("COMPUTATIONAL EFFICIENCY ANALYSIS (YELLOW FRAME & Z24 BRIDGE)")
    print("=" * 160)

    for dataset_name, models_dict in DATASET_MODEL_CONFIGS.items():
        print(f"\n>>> Dataset: {dataset_name}")
        dataset_rows = []

        for model_label, cfg in models_dict.items():
            target_w = SELECTED_WEIGHTS[dataset_name][model_label]
            all_studies = discover_all_studies(cfg)
            target_study = select_target_study(all_studies, target_w)
            if target_study is None:
                print(f"  [WARNING] No .db file found for {dataset_name} | {model_label} (weight={target_w})")
                continue

            res = analyze_study_efficiency(
                dataset_name, model_label, cfg["type"], target_study, all_studies, n_folds=N_FOLDS
            )
            if res is not None:
                dataset_rows.append(res)
                all_rows.append(res)
                print(
                    f"  - {model_label:15s} | w={res['Weight']:5s} | "
                    f"BS={res['Avg. Batch Size']:11s} | "
                    f"Epoch={res['Time/Epoch (s)']:11s} s | "
                    f"Fold={res['Time/Fold (min)']:11s} min | "
                    f"Comp={res['Time/Completed Trial (min)']:11s} min | "
                    f"PrunedT={res['Time/Pruned Trial (min)']:11s} min | "
                    f"Pruned={res['Pruned Trials N (%)']:12s} | "
                    f"MeanPruned(All w)={res['Mean Pruned Trials (All w)']:18s} | "
                    f"Source={res['_epoch_source']}"
                )

        if dataset_rows:
            mean_row = build_dataset_mean_row(dataset_name, dataset_rows)
            all_rows.append(mean_row)

    if not all_rows:
        print("\nNo Optuna studies were processed. Verify BASE_DIR_OUTPUT and directory names.")
        return

    df_full = pd.DataFrame(all_rows)
    table_cols = [
        "Dataset",
        "Model",
        "Weight",
        "Params (x10^3)",
        "Avg. Batch Size",
        "Time/Epoch (s)",
        "Time/Fold (min)",
        "Time/Completed Trial (min)",
        "Time/Pruned Trial (min)",
        "Pruned Trials N (%)",
        "Mean Pruned Trials (All w)",
        "Total HPO (h)",
    ]
    df_table = df_full[table_cols]

    print("\n" + "=" * 180)
    print("UNIFIED COMPUTATIONAL EFFICIENCY TABLE")
    print("=" * 180)
    print(df_table.to_string(index=False))

    csv_path = os.path.join(RESULTS_SAVE_DIR, "computational_efficiency_table.csv")
    xlsx_path = os.path.join(RESULTS_SAVE_DIR, "computational_efficiency_table.xlsx")
    tex_path = os.path.join(RESULTS_SAVE_DIR, "computational_efficiency_table.tex")

    df_table.to_csv(csv_path, index=False)
    df_table.to_excel(xlsx_path, index=False, sheet_name="Computational_Efficiency")
    export_latex_table(df_full, tex_path)

    print(
        f"\nSaved files:\n"
        f"  - {csv_path}\n"
        f"  - {xlsx_path}\n"
        f"  - {tex_path}"
    )


if __name__ == "__main__":
    main()