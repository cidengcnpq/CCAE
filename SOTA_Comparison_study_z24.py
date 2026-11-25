import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import mahalanobis
from scipy.stats import mannwhitneyu
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Suppress non-critical warnings and Optuna info logs
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# 1. CONFIGURATION SECTION (Z24 BRIDGE BENCHMARK)
# ==============================================================================
BASE_DIR_OUTPUT = r"C:\Users\Victor Higino\Documents\Doutorado\CCAE"
ROOT_DATA_DIR = r"C:\Users\Victor Higino\Documents\Doutorado\CCAE\Imagens_CWT_Z24_v1_matlab"
RESULTS_SAVE_DIR = os.path.join(BASE_DIR_OUTPUT, "comparison_results_z24")

# Default sensor identifiers on the Z24 Bridge girder (Bern side).
# Automatically falls back to Sensor1..Sensor7 if folders are named generically.
DEFAULT_Z24_SENSORS = ["105V", "110V", "115V", "120V", "125V", "130V", "135V"]
FALLBACK_GENERIC_SENSORS = [f"Sensor{i}" for i in range(1, 8)]
NUM_SENSORS = 7
N_FOLDS = 4

# All 7 scenarios are processed for boxplots and Mann-Whitney U hypothesis tests;
# ONLY TEST_SCENARIO_LABELS are evaluated in the localization accuracy tables.
DAMAGE_SCENARIOS = [
    "d_0_intact",
    "d_0_unknown",
    "d_1",
    "d_2",
    "d_4",
    "d_5",
]
INTACT_CONDITION_NAME = "d_0_intact"

SCENARIO_LABELS = {
    "d_0_intact": "Intact (train)",
    "d_0_unknown": "Intact (test)",
    "d_1": "Damage 1",
    "d_2": "Damage 2",
    "d_3": "Damage 3",
    "d_4": "Damage 4",
    "d_5": "Damage 5",
}

TEST_SCENARIO_LABELS = [
    "Intact (test)",
    "Damage 1",
    "Damage 2",
    "Damage 3",
    "Damage 4",
    "Damage 5",
]

GROUP_ORDER = ["g1", "g2", "g3", "g4", "g5", "g6"]

# Scenario-specific damaged groups for progressive pier settlement on Z24 Bridge:
# Scenario 1 (d_1): ["g6"]
# Scenario 2 (d_2): ["g5", "g6"]
# Scenario 3 (d_3): ["g4", "g5", "g6"]
# Scenario 4 (d_4): ["g4", "g5", "g6"]
# Scenario 5 (d_5): ["g4", "g5", "g6"]
DAMAGED_GROUPS_FOR_SETTLEMENT = {
    "Intact (test)": [],
    "Damage 1": ["g4", "g5", "g6"],
    "Damage 2": ["g4", "g5", "g6"],
    "Damage 3": ["g4", "g5", "g6"],
    "Damage 4": ["g4", "g5", "g6"],
    "Damage 5": ["g4", "g5", "g6"],
}

EXPECTED_LOCALIZATION_STATE = {
    scen: {
        g: ("Above" if g in DAMAGED_GROUPS_FOR_SETTLEMENT[scen] else "Below")
        for g in GROUP_ORDER
    }
    for scen in TEST_SCENARIO_LABELS
}

# Percentile used on Intact (train) to define the per-group threshold
THRESHOLD_PERCENTILE = 99

FIXED_PARAMS = {
    "classifier_linear_dim": 64,
    "bottleneck_channels": 4,
    "kl_sparsity_rho": 0.05,
}

# Criterion to choose the best weight configuration per model family:
# "best_fold_loss"  -> selects the weight that achieved the lowest loss on the best fold
# "val_loss"        -> selects the weight with the lowest Optuna cross-validation loss
# "global_accuracy" -> selects the weight with highest accuracy (tie-broken by lowest loss)
BEST_WEIGHT_SELECTION_METRIC = "global_accuracy"

# Statistical significance level (alpha) for the one-sided Mann-Whitney U test
ALPHA_SIGNIFICANCE = 0.05

# Keep boxplots clean with the threshold line only (without hypothesis test stars)
ANNOTATE_SIGNIFICANCE_ON_BOXPLOTS = False

MODEL_CONFIGS = {
    "CCAE": {
        "type": "ccae",
        "output_dir_candidates": [
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24"),
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_bridge"),
        ],
        "study_name": "ccae_hyperparam_optimization_z24",
        "db_prefix": "ccae_optimization_z24",
        "multi_weight": True,
        "weights": [1e-3, 1e-2, 1e-1],
    },
    "Traditional CAE": {
        "type": "cae",
        "output_dir_candidates": [
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_CAE"),
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_bridge_CAE"),
        ],
        "study_name": "cae_hyperparam_optimization_z24",
        "db_prefix": "cae_optimization_z24",
        "multi_weight": False,
        "fixed_weight_label": "N/A",
        "fixed_weight_value": 0.0,
    },
    "VAE": {
        "type": "vae",
        "output_dir_candidates": [
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_VAE"),
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_bridge_VAE"),
        ],
        "study_name": "vae_hyperparam_optimization_z24",
        "db_prefix": "vae_optimization_z24",
        "multi_weight": True,
        "weights": [1e-3, 1e-2, 1e-1],
    },
    "SAE-L1": {
        "type": "sae_l1",
        "output_dir_candidates": [
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_SAE_L1"),
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_bridge_SAE_L1"),
        ],
        "study_name": "sae_l1_hyperparam_optimization_z24",
        "db_prefix": "sae_l1_optimization_z24",
        "multi_weight": True,
        "weights": [1e-3, 1e-2, 1e-1],
    },
    "SAE-KL": {
        "type": "sae_kl",
        "output_dir_candidates": [
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_SAE_kl"),
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_SAE_KL"),
            os.path.join(BASE_DIR_OUTPUT, "optuna_optimization_output_z24_bridge_SAE_kl"),
        ],
        "study_name": "sae_kl_hyperparam_optimization_z24",
        "db_prefix": "sae_kl_optimization_z24",
        "multi_weight": True,
        "weights": [1e-3, 1e-2, 1e-1],
    },
}


# ==============================================================================
# 2. MODEL ARCHITECTURES (CCAE, Traditional CAE, VAE, SAE-L1, SAE-KL)
# ==============================================================================
def build_shared_encoder():
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    )


def build_shared_decoder(decoder_input_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(decoder_input_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid(),
    )


class CCAE(nn.Module):
    def __init__(self, num_sensors, bottleneck_channels=4, classifier_linear_dim=64, dropout_rate=0.1):
        super(CCAE, self).__init__()
        self.num_sensors = num_sensors
        self.encoder = build_shared_encoder()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
        )
        self.sensor_embedding = nn.Embedding(num_sensors, 64)
        classifier_input_dim = bottleneck_channels * 8 * 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_input_dim, classifier_linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_linear_dim, num_sensors),
        )
        self.decoder = build_shared_decoder(bottleneck_channels + 1)

    def forward(self, x):
        x_enc = self.encoder(x)
        encoded_features = self.bottleneck_conv(x_enc)
        sensor_logits = self.classifier(encoded_features)
        sensor_probs = torch.softmax(sensor_logits, dim=1)
        all_emb = self.sensor_embedding(torch.arange(self.num_sensors, device=x.device))
        combined_emb = (sensor_probs.unsqueeze(2) * all_emb.unsqueeze(0)).sum(dim=1)
        combined_emb_reshaped = combined_emb.view(-1, 1, 8, 8)
        concat_features = torch.cat([encoded_features, combined_emb_reshaped], dim=1)
        return self.decoder(concat_features), sensor_logits

    def get_latent_features(self, x, scaling_factor=1.0):
        with torch.no_grad():
            x_enc = self.encoder(x)
            encoded_features = self.bottleneck_conv(x_enc)
            sensor_logits = self.classifier(encoded_features)
            sensor_probs = torch.softmax(sensor_logits, dim=1)
            all_emb = self.sensor_embedding(torch.arange(self.num_sensors, device=x.device))
            combined_emb = (sensor_probs.unsqueeze(2) * all_emb.unsqueeze(0)).sum(dim=1)
            combined_emb_reshaped = combined_emb.view(-1, 1, 8, 8) * scaling_factor
            concat_features = torch.cat([encoded_features, combined_emb_reshaped], dim=1)
            pooled = nn.AdaptiveAvgPool2d((1, 1))(concat_features)
            return pooled.view(pooled.size(0), -1)


class TraditionalCAE(nn.Module):
    def __init__(self, bottleneck_channels=4):
        super(TraditionalCAE, self).__init__()
        self.encoder = build_shared_encoder()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
        )
        self.decoder = build_shared_decoder(bottleneck_channels)

    def forward(self, x):
        encoded_features = self.bottleneck_conv(self.encoder(x))
        return self.decoder(encoded_features)

    def get_latent_features(self, x, scaling_factor=1.0):
        with torch.no_grad():
            encoded_features = self.bottleneck_conv(self.encoder(x))
            pooled = nn.AdaptiveAvgPool2d((1, 1))(encoded_features)
            return pooled.view(pooled.size(0), -1)


class ConvSAE(nn.Module):
    def __init__(self, bottleneck_channels=4, sparsity_mode="l1"):
        super(ConvSAE, self).__init__()
        self.encoder = build_shared_encoder()
        bottleneck_act = nn.Sigmoid() if sparsity_mode == "kl" else nn.ReLU()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            bottleneck_act,
        )
        self.decoder = build_shared_decoder(bottleneck_channels)

    def forward(self, x):
        encoded_features = self.bottleneck_conv(self.encoder(x))
        return self.decoder(encoded_features), encoded_features

    def get_latent_features(self, x, scaling_factor=1.0):
        with torch.no_grad():
            encoded_features = self.bottleneck_conv(self.encoder(x))
            pooled = nn.AdaptiveAvgPool2d((1, 1))(encoded_features)
            return pooled.view(pooled.size(0), -1)


class ConvVAE(nn.Module):
    def __init__(self, bottleneck_channels=4, mu_layer_name="conv_mu", logvar_layer_name="conv_logvar"):
        super(ConvVAE, self).__init__()
        self.encoder = build_shared_encoder()
        self.mu_layer_name = mu_layer_name
        self.logvar_layer_name = logvar_layer_name
        setattr(self, mu_layer_name, nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1))
        setattr(self, logvar_layer_name, nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1))
        self.decoder = build_shared_decoder(bottleneck_channels)

    def forward(self, x):
        h = self.encoder(x)
        mu = getattr(self, self.mu_layer_name)(h)
        logvar = torch.clamp(getattr(self, self.logvar_layer_name)(h), min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return self.decoder(z), mu, logvar

    def get_latent_features(self, x, scaling_factor=1.0):
        with torch.no_grad():
            h = self.encoder(x)
            mu = getattr(self, self.mu_layer_name)(h)
            pooled = nn.AdaptiveAvgPool2d((1, 1))(mu)
            return pooled.view(pooled.size(0), -1)


# ==============================================================================
# 3. LOSS DECOMPOSITION FUNCTIONS (SSIM RECONSTRUCTION + PENALTY + TOTAL LOSS)
# ==============================================================================
def _create_ssim_window(window_size=11, sigma=1.5, channel=1, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def compute_ssim_loss_batch(img1, img2, window_size=11):
    """Computes SSIM reconstruction loss (1 - SSIM) in pure PyTorch."""
    channel = img1.size(1)
    window = _create_ssim_window(window_size=window_size, channel=channel, device=img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return 1.0 - ssim_map.mean()


def evaluate_fold_loss_decomposition(model, model_type, weight_val, best_params, all_sensor_data, sensors_list, device):
    """
    Evaluates the model on the intact baseline set (d_0_intact) across all sensors and returns:
      - recon_loss: SSIM Reconstruction Loss
      - penalty_raw: Unweighted regularization/classification penalty
      - penalty_weighted: weight_val * penalty_raw
      - total_loss: recon_loss + penalty_weighted
    """
    model.eval()
    ce_criterion = nn.CrossEntropyLoss()
    rho = float(best_params.get("sparsity_target", FIXED_PARAMS["kl_sparsity_rho"])) if best_params else FIXED_PARAMS["kl_sparsity_rho"]

    total_samples = 0
    sum_recon = 0.0
    sum_pen_raw = 0.0

    with torch.no_grad():
        for sensor_idx, sensor_name in enumerate(sensors_list):
            dataloader = all_sensor_data.get(sensor_name, {}).get(INTACT_CONDITION_NAME, None)
            if dataloader is None:
                continue
            for images, _ in dataloader:
                if images.numel() == 0:
                    continue
                images = images.to(device)
                bsz = images.size(0)

                if model_type == "ccae":
                    recon, logits = model(images)
                    r_loss = compute_ssim_loss_batch(images, recon).item()
                    targets = torch.full((bsz,), sensor_idx, dtype=torch.long, device=device)
                    p_raw = ce_criterion(logits, targets).item()
                elif model_type == "cae":
                    recon = model(images)
                    r_loss = compute_ssim_loss_batch(images, recon).item()
                    p_raw = 0.0
                elif model_type == "vae":
                    recon, mu, logvar = model(images)
                    r_loss = compute_ssim_loss_batch(images, recon).item()
                    p_raw = float(-0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp()).item())
                elif model_type == "sae_l1":
                    recon, z = model(images)
                    r_loss = compute_ssim_loss_batch(images, recon).item()
                    p_raw = float(torch.mean(torch.abs(z)).item())
                elif model_type == "sae_kl":
                    recon, z = model(images)
                    r_loss = compute_ssim_loss_batch(images, recon).item()
                    rho_hat = torch.clamp(torch.mean(z, dim=0), min=1e-6, max=1.0 - 1e-6)
                    kl_div = rho * torch.log(rho / rho_hat) + (1.0 - rho) * torch.log((1.0 - rho) / (1.0 - rho_hat))
                    p_raw = float(torch.mean(kl_div).item())
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                sum_recon += r_loss * bsz
                sum_pen_raw += p_raw * bsz
                total_samples += bsz

    if total_samples == 0:
        return {"recon_loss": np.nan, "penalty_raw": np.nan, "penalty_weighted": np.nan, "total_loss": np.nan}

    avg_recon = sum_recon / total_samples
    avg_pen_raw = sum_pen_raw / total_samples
    avg_pen_weighted = weight_val * avg_pen_raw
    avg_total = avg_recon + avg_pen_weighted

    return {
        "recon_loss": float(avg_recon),
        "penalty_raw": float(avg_pen_raw),
        "penalty_weighted": float(avg_pen_weighted),
        "total_loss": float(avg_total),
    }


# ==============================================================================
# 4. DATASET, SENSOR GROUP RESOLUTION, AND OPTUNA 4-FOLD DISCOVERY
# ==============================================================================
class CWTDatasetAnalysis(Dataset):
    def __init__(self, root_folder):
        self.image_paths = sorted(glob.glob(os.path.join(root_folder, "**", "*.png"), recursive=True))
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path)
            return self.transform(img), os.path.basename(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None


def collate_fn_analysis(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return torch.tensor([]), []
    images, names = zip(*batch)
    return torch.stack(images), list(names)


def resolve_sensor_names_and_groups(root_dir, intact_condition):
    """
    Automatically checks whether the Z24 sensor folders are named
    ['105V', '110V', ..., '135V'] or ['Sensor1', ..., 'Sensor7']
    and builds the 6 overlapping sensor pairs (g1..g6).
    """
    intact_dir = os.path.join(root_dir, intact_condition)
    if os.path.isdir(intact_dir):
        subdirs = sorted([
            d for d in os.listdir(intact_dir)
            if os.path.isdir(os.path.join(intact_dir, d))
        ])
        if all(s in subdirs for s in DEFAULT_Z24_SENSORS):
            sensors = DEFAULT_Z24_SENSORS
        elif all(s in subdirs for s in FALLBACK_GENERIC_SENSORS):
            sensors = FALLBACK_GENERIC_SENSORS
        elif len(subdirs) == NUM_SENSORS:
            sensors = subdirs
        else:
            sensors = DEFAULT_Z24_SENSORS
    else:
        sensors = DEFAULT_Z24_SENSORS

    sensor_groups = {
        f"g{i + 1}": [sensors[i], sensors[i + 1]]
        for i in range(len(sensors) - 1)
    }
    return sensors, sensor_groups


def load_all_sensor_data(root_dir, damage_scenarios, sensors_list):
    all_data = {sensor: {} for sensor in sensors_list}
    for sensor_name in sensors_list:
        for condition in damage_scenarios:
            path = os.path.join(root_dir, condition, sensor_name)
            if os.path.isdir(path):
                dataset = CWTDatasetAnalysis(path)
                if len(dataset) > 0:
                    all_data[sensor_name][condition] = DataLoader(
                        dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_analysis
                    )
            else:
                print(f"Warning: Directory not found: {path}")
    return all_data


def _resolve_output_dir(cfg):
    for candidate in cfg.get("output_dir_candidates", []):
        if os.path.isdir(candidate):
            return candidate
    return cfg["output_dir_candidates"][0]


def _locate_all_fold_checkpoints(trial_dir, n_folds=N_FOLDS):
    """Finds all 4 fold checkpoints (best_model_fold_1..4.pth) inside a trial folder."""
    fold_ckpts = {}
    for fold_idx in range(1, n_folds + 1):
        candidate = os.path.join(trial_dir, f"best_model_fold_{fold_idx}.pth")
        if os.path.isfile(candidate):
            fold_ckpts[fold_idx] = candidate

    if not fold_ckpts:
        any_pth = sorted(glob.glob(os.path.join(trial_dir, "*.pth")))
        for idx, pth in enumerate(any_pth[:n_folds], start=1):
            fold_ckpts[idx] = pth

    return fold_ckpts


def find_all_weight_models_from_config(model_label, cfg):
    """
    Returns a list of all trained weight configurations for a model family.
    Each item contains all available fold checkpoints (up to 4 folds) of the best Optuna trial.
    """
    output_dir = _resolve_output_dir(cfg)
    if not os.path.isdir(output_dir):
        print(f"[{model_label}] Output directory not found: {output_dir}")
        return []

    found_variations = []

    if not cfg.get("multi_weight", False):
        db_files = glob.glob(os.path.join(output_dir, "*.db"))
        db_path = os.path.join(output_dir, f"{cfg['db_prefix']}.db")
        if not os.path.isfile(db_path) and db_files:
            db_path = db_files[0]
        if os.path.isfile(db_path):
            storage_url = f"sqlite:///{db_path}"
            summaries = optuna.get_all_study_summaries(storage=storage_url)
            study_name = summaries[0].study_name if summaries else cfg["study_name"]
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            trial_dir = os.path.join(output_dir, f"trial_{study.best_trial.number}")
            fold_ckpts = _locate_all_fold_checkpoints(trial_dir)
            if fold_ckpts:
                found_variations.append({
                    "weight_label": cfg.get("fixed_weight_label", "N/A"),
                    "weight_value": float(cfg.get("fixed_weight_value", 0.0)),
                    "val_loss": float(study.best_trial.value),
                    "trial_number": study.best_trial.number,
                    "fold_ckpts": fold_ckpts,
                    "best_params": study.best_trial.params,
                })
    else:
        for w in cfg.get("weights", [1e-3, 1e-2, 1e-1]):
            weight_tag = f"{w:.0e}".replace("+", "")
            sub_dir = os.path.join(output_dir, f"weight_{weight_tag}")
            if not os.path.isdir(sub_dir):
                print(f"  [{model_label}] Weight subfolder not found: {sub_dir}")
                continue
            db_files = glob.glob(os.path.join(sub_dir, "*.db"))
            if not db_files:
                continue
            storage_url = f"sqlite:///{db_files[0]}"
            summaries = optuna.get_all_study_summaries(storage=storage_url)
            if not summaries:
                continue
            study = optuna.load_study(study_name=summaries[0].study_name, storage=storage_url)
            if not study.best_trials:
                continue
            trial_dir = os.path.join(sub_dir, f"trial_{study.best_trial.number}")
            fold_ckpts = _locate_all_fold_checkpoints(trial_dir)
            if fold_ckpts:
                found_variations.append({
                    "weight_label": weight_tag,
                    "weight_value": float(w),
                    "val_loss": float(study.best_trial.value),
                    "trial_number": study.best_trial.number,
                    "fold_ckpts": fold_ckpts,
                    "best_params": study.best_trial.params,
                })

    return found_variations


def instantiate_and_load_model(cfg, ckpt_path, best_params, device):
    model_type = cfg["type"]
    state_dict = torch.load(ckpt_path, map_location=device)

    if model_type == "ccae":
        dropout = best_params.get("dropout_rate", 0.1) if best_params else 0.1
        model = CCAE(
            num_sensors=NUM_SENSORS,
            bottleneck_channels=FIXED_PARAMS["bottleneck_channels"],
            classifier_linear_dim=FIXED_PARAMS["classifier_linear_dim"],
            dropout_rate=dropout,
        )
    elif model_type == "cae":
        model = TraditionalCAE(bottleneck_channels=FIXED_PARAMS["bottleneck_channels"])
    elif model_type == "sae_l1":
        model = ConvSAE(bottleneck_channels=FIXED_PARAMS["bottleneck_channels"], sparsity_mode="l1")
    elif model_type == "sae_kl":
        model = ConvSAE(bottleneck_channels=FIXED_PARAMS["bottleneck_channels"], sparsity_mode="kl")
    elif model_type == "vae":
        non_enc_dec_prefixes = sorted({
            k.split(".")[0] for k in state_dict.keys()
            if not k.startswith("encoder.") and not k.startswith("decoder.")
        })
        mu_name = next((p for p in non_enc_dec_prefixes if "mu" in p.lower()), "conv_mu")
        logvar_name = next((p for p in non_enc_dec_prefixes if "var" in p.lower() or "sigma" in p.lower()), "conv_logvar")
        model = ConvVAE(
            bottleneck_channels=FIXED_PARAMS["bottleneck_channels"],
            mu_layer_name=mu_name,
            logvar_layer_name=logvar_name,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ==============================================================================
# 5. HIERARCHICAL ANOMALY DETECTION PIPELINE (PER FOLD)
# ==============================================================================
def calculate_scaling_factor_ccae(model, intact_dataloaders, device):
    """Calculates the Frobenius norm scaling factor exclusively for CCAE."""
    model.eval()
    encoded_norms, sensor_emb_norms = [], []
    with torch.no_grad():
        for dataloader in intact_dataloaders:
            for images, _ in dataloader:
                if images.numel() == 0:
                    continue
                images = images.to(device)
                x_enc = model.encoder(images)
                encoded_features = model.bottleneck_conv(x_enc)
                logits = model.classifier(encoded_features)
                probs = torch.softmax(logits, dim=1)
                all_emb = model.sensor_embedding(torch.arange(model.num_sensors, device=device))
                combined_emb = (probs.unsqueeze(2) * all_emb.unsqueeze(0)).sum(dim=1).view(-1, 1, 8, 8)

                encoded_norms.append(torch.linalg.norm(encoded_features.flatten(start_dim=1), dim=1).cpu().numpy())
                sensor_emb_norms.append(torch.linalg.norm(combined_emb.flatten(start_dim=1), dim=1).cpu().numpy())

    if not encoded_norms or not sensor_emb_norms:
        return 1.0
    avg_enc = np.mean(np.concatenate(encoded_norms))
    avg_emb = np.mean(np.concatenate(sensor_emb_norms))
    return float(avg_enc / avg_emb) if avg_emb != 0 else 1.0


def evaluate_single_fold_hierarchy(
    model_label,
    weight_label,
    fold_idx,
    model,
    model_type,
    all_sensor_data,
    sensors_list,
    sensor_groups_map,
    device,
):
    """
    Runs feature extraction, per-sensor Mahalanobis distance, and group-level (g1..g6)
    KDE fusion for a single fold checkpoint on the Z24 Bridge dataset.
    """
    scaling_factor = 1.0
    if model_type == "ccae":
        intact_loaders = [
            all_sensor_data[s][INTACT_CONDITION_NAME]
            for s in sensors_list
            if INTACT_CONDITION_NAME in all_sensor_data[s]
        ]
        scaling_factor = calculate_scaling_factor_ccae(model, intact_loaders, device)

    # 1. Extract latent features per sensor and scenario
    latent_features = {s: {c: {} for c in DAMAGE_SCENARIOS} for s in sensors_list}
    for s in sensors_list:
        for c, dataloader in all_sensor_data[s].items():
            with torch.no_grad():
                for images, names in dataloader:
                    if images.numel() == 0:
                        continue
                    feats = model.get_latent_features(images.to(device), scaling_factor).cpu().numpy()
                    for idx, name in enumerate(names):
                        latent_features[s][c][name] = feats[idx]

    # 2. Sensor-level Mahalanobis Distance
    sensor_md = {s: {c: {} for c in DAMAGE_SCENARIOS} for s in sensors_list}
    for s in sensors_list:
        intact_dict = latent_features[s].get(INTACT_CONDITION_NAME, {})
        if len(intact_dict) <= 1:
            continue
        intact_arr = np.array(list(intact_dict.values()))
        variances = np.var(intact_arr, axis=0)
        valid_idx = np.where(variances > 1e-9)[0]
        if valid_idx.size == 0:
            continue

        intact_filtered = intact_arr[:, valid_idx]
        mean_vec = np.mean(intact_filtered, axis=0)
        cov = EmpiricalCovariance().fit(intact_filtered).covariance_
        inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

        for c in DAMAGE_SCENARIOS:
            cond_dict = latent_features[s].get(c, {})
            for fname, feat in cond_dict.items():
                f_filt = feat[valid_idx]
                sensor_md[s][c][fname] = mahalanobis(f_filt, mean_vec, inv_cov)

    # 3. System-level Sensor Group (g1..g6) KDE Fusion
    df_rows = []
    for grp_name, sensors_in_grp in sensor_groups_map.items():
        intact_fnames = sorted(
            set.intersection(*(set(sensor_md[s][INTACT_CONDITION_NAME].keys()) for s in sensors_in_grp))
        )
        intact_mds = np.array([
            [sensor_md[s][INTACT_CONDITION_NAME][fn] for s in sensors_in_grp]
            for fn in intact_fnames
        ])
        if intact_mds.shape[0] < 5:
            continue

        grid = GridSearchCV(
            KernelDensity(kernel="gaussian"),
            {"bandwidth": 10 ** np.linspace(-2, 2, 100)},
            cv=min(5, intact_mds.shape[0]),
        )
        grid.fit(intact_mds)
        kde_grp = grid.best_estimator_

        for c in DAMAGE_SCENARIOS:
            cond_fnames = sorted(
                set.intersection(*(set(sensor_md[s][c].keys()) for s in sensors_in_grp))
            )
            for fn in cond_fnames:
                md_vec = np.array([[sensor_md[s][c][fn] for s in sensors_in_grp]])
                di_kde = float(-kde_grp.score_samples(md_vec)[0])
                if di_kde > 0:
                    df_rows.append({
                        "Model": model_label,
                        "Weight": weight_label,
                        "Fold": fold_idx,
                        "Group": grp_name,
                        "Scenario": SCENARIO_LABELS[c],
                        "Filename": fn,
                        "DI": di_kde,
                    })

    df_fold = pd.DataFrame(df_rows)
    if df_fold.empty:
        return df_fold, {}

    df_fold["Group"] = pd.Categorical(df_fold["Group"], categories=GROUP_ORDER, ordered=True)
    df_fold["Scenario"] = pd.Categorical(
        df_fold["Scenario"], categories=list(SCENARIO_LABELS.values()), ordered=True
    )
    df_fold = df_fold.sort_values(by=["Group", "Scenario"]).reset_index(drop=True)

    # 4. Percentile threshold per sensor group computed on Intact (train)
    group_thresholds = {}
    for grp_name in GROUP_ORDER:
        intact_vals = df_fold[
            (df_fold["Group"] == grp_name) & (df_fold["Scenario"] == "Intact (train)")
        ]["DI"]
        if not intact_vals.empty:
            group_thresholds[grp_name] = float(np.percentile(intact_vals, THRESHOLD_PERCENTILE))

    return df_fold, group_thresholds


# ==============================================================================
# 6. 4-FOLD & BEST-FOLD ACCURACY AND LOSS METRICS (EXCLUDING TRAIN SET)
# ==============================================================================
def compute_single_fold_accuracy(model_label, weight_label, fold_idx, df_fold, group_thresholds, loss_dict):
    """Computes group-level and scenario-level accuracies for a single fold (test scenarios only)."""
    group_rows = []
    df_test_only = df_fold[df_fold["Scenario"].isin(TEST_SCENARIO_LABELS)]

    for (grp_name, scenario), group in df_test_only.groupby(["Group", "Scenario"], observed=False):
        if grp_name not in group_thresholds or len(group) == 0:
            continue

        thr = group_thresholds[grp_name]
        total = len(group)
        n_below = int((group["DI"] <= thr).sum())
        n_above = int((group["DI"] > thr).sum())

        pct_below = (n_below / total) * 100.0
        pct_above = (n_above / total) * 100.0

        expected_pos = EXPECTED_LOCALIZATION_STATE[scenario][grp_name]
        loc_accuracy = pct_below if expected_pos == "Below" else pct_above

        group_rows.append({
            "Model": model_label,
            "Weight": weight_label,
            "Fold": fold_idx,
            "Scenario": scenario,
            "Sensor_Group": grp_name,
            "Expected_Position": f"{expected_pos} Threshold",
            "Threshold": thr,
            "Total_Samples": total,
            "Samples_Below (<=Thr)": n_below,
            "Samples_Above (>Thr)": n_above,
            "Pct_Below (%)": pct_below,
            "Pct_Above (%)": pct_above,
            "Localization_Accuracy (%)": loc_accuracy,
            "Recon_Loss": loss_dict["recon_loss"],
            "Penalty_Raw": loss_dict["penalty_raw"],
            "Penalty_Weighted": loss_dict["penalty_weighted"],
            "Total_Loss": loss_dict["total_loss"],
        })

    df_groups_fold = pd.DataFrame(group_rows)
    if df_groups_fold.empty:
        return df_groups_fold, {}

    acc_intact_test = df_groups_fold[df_groups_fold["Scenario"] == "Intact (test)"]["Localization_Accuracy (%)"].mean()

    dmg_scenarios = ["Damage 1", "Damage 2", "Damage 3", "Damage 4", "Damage 5"]
    fold_summary = {
        "Fold": fold_idx,
        "Recon_Loss": loss_dict["recon_loss"],
        "Penalty_Raw": loss_dict["penalty_raw"],
        "Penalty_Weighted": loss_dict["penalty_weighted"],
        "Total_Loss": loss_dict["total_loss"],
        "Intact (test)": acc_intact_test,
    }

    for d_scen in dmg_scenarios:
        sub_d = df_groups_fold[df_groups_fold["Scenario"] == d_scen]
        fold_summary[f"{d_scen} (Damaged > Thr)"] = sub_d[
            sub_d["Expected_Position"] == "Above Threshold"
        ]["Localization_Accuracy (%)"].mean()
        fold_summary[f"{d_scen} (Undamaged <= Thr)"] = sub_d[
            sub_d["Expected_Position"] == "Below Threshold"
        ]["Localization_Accuracy (%)"].mean()
        fold_summary[f"{d_scen} (Overall)"] = sub_d["Localization_Accuracy (%)"].mean()

    sub_all_dmg = df_groups_fold[df_groups_fold["Scenario"].isin(dmg_scenarios)]
    fold_summary["All Damages - Settlement Zone (> Thr)"] = sub_all_dmg[
        sub_all_dmg["Expected_Position"] == "Above Threshold"
    ]["Localization_Accuracy (%)"].mean()
    fold_summary["All Damages - Undamaged Zone (<= Thr)"] = sub_all_dmg[
        sub_all_dmg["Expected_Position"] == "Below Threshold"
    ]["Localization_Accuracy (%)"].mean()
    fold_summary["Global Accuracy"] = df_groups_fold["Localization_Accuracy (%)"].mean()

    return df_groups_fold, fold_summary


def format_mean_std(values, decimals=2):
    """Formats an array of fold values as 'Mean ± Std'."""
    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean_val, std_val, f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def summarize_4fold_and_best_fold_metrics(
    model_label, weight_label, val_loss, fold_group_dfs, fold_summaries
):
    """
    Identifies the BEST FOLD (lowest Total_Loss) and aggregates both:
      1) 4-Fold Mean ± Std (%) AND Best Fold Accuracy (%) per group and scenario
      2) Separate Loss Decomposition (Recon Loss, Penalty Raw, Penalty Weighted, Total Loss)
    """
    df_all_folds_groups = pd.concat(fold_group_dfs, ignore_index=True)
    df_fold_sum = pd.DataFrame(fold_summaries)

    # Identify the best fold as the fold with the lowest Total_Loss
    best_fold_row = df_fold_sum.sort_values(by=["Total_Loss", "Global Accuracy"], ascending=[True, False]).iloc[0]
    best_fold_idx = int(best_fold_row["Fold"])

    # 1. Group-level aggregation across the 4 folds + Best Fold value (g1..g6)
    group_agg_rows = []
    for (scenario, grp_name, exp_pos), grp in df_all_folds_groups.groupby(
        ["Scenario", "Sensor_Group", "Expected_Position"], observed=False
    ):
        acc_mean, acc_std, acc_str = format_mean_std(grp["Localization_Accuracy (%)"].values, decimals=2)
        thr_mean, thr_std, thr_str = format_mean_std(grp["Threshold"].values, decimals=4)

        best_grp_row = grp[grp["Fold"] == best_fold_idx]
        best_fold_acc = float(best_grp_row["Localization_Accuracy (%)"].iloc[0]) if not best_grp_row.empty else np.nan

        group_agg_rows.append({
            "Model": model_label,
            "Weight": weight_label,
            "Best_Fold": best_fold_idx,
            "Scenario": scenario,
            "Sensor_Group": grp_name,
            "Expected_Position": exp_pos,
            "Threshold (4-Fold Mean ± Std)": thr_str,
            "Accuracy_Mean (%)": round(acc_mean, 2),
            "Accuracy_Std (%)": round(acc_std, 2),
            "Best_Fold_Accuracy (%)": round(best_fold_acc, 2),
            "Localization_Accuracy (4-Fold Mean ± Std %)": acc_str,
            "Combined_4Fold_and_BestFold (%)": f"{acc_str} (Best F{best_fold_idx}: {best_fold_acc:.2f}%)",
        })
    df_group_agg = pd.DataFrame(group_agg_rows)

    # 2. Loss decomposition across the 4 folds + Best Fold
    rec_m, rec_s, rec_str = format_mean_std(df_fold_sum["Recon_Loss"], decimals=5)
    pen_r_m, pen_r_s, pen_r_str = format_mean_std(df_fold_sum["Penalty_Raw"], decimals=5)
    pen_w_m, pen_w_s, pen_w_str = format_mean_std(df_fold_sum["Penalty_Weighted"], decimals=5)
    tot_m, tot_s, tot_str = format_mean_std(df_fold_sum["Total_Loss"], decimals=5)

    loss_row = {
        "Model": model_label,
        "Weight": weight_label,
        "Folds_Evaluated": len(fold_summaries),
        "Optuna_CV_Loss": round(val_loss, 5),
        "Recon_Loss_SSIM (4-Fold Mean ± Std)": rec_str,
        "Penalty_Raw (4-Fold Mean ± Std)": pen_r_str,
        "Penalty_Weighted (4-Fold Mean ± Std)": pen_w_str,
        "Total_Loss (4-Fold Mean ± Std)": tot_str,
        "Best_Fold (Lowest Loss)": f"Fold {best_fold_idx}",
        "Best_Fold_Recon_Loss": round(float(best_fold_row["Recon_Loss"]), 5),
        "Best_Fold_Penalty_Raw": round(float(best_fold_row["Penalty_Raw"]), 5),
        "Best_Fold_Penalty_Weighted": round(float(best_fold_row["Penalty_Weighted"]), 5),
        "Best_Fold_Total_Loss": round(float(best_fold_row["Total_Loss"]), 5),
    }

    # 3. Scenario & Global accuracy summary (4-Fold Mean ± Std AND Best Fold)
    intact_m, _, intact_str = format_mean_std(df_fold_sum["Intact (test)"])
    d1_m, _, d1_str = format_mean_std(df_fold_sum["Damage 1 (Overall)"])
    d2_m, _, d2_str = format_mean_std(df_fold_sum["Damage 2 (Overall)"])
    d3_m, _, d3_str = format_mean_std(df_fold_sum["Damage 3 (Overall)"])
    d4_m, _, d4_str = format_mean_std(df_fold_sum["Damage 4 (Overall)"])
    d5_m, _, d5_str = format_mean_std(df_fold_sum["Damage 5 (Overall)"])

    kop_m, _, kop_str = format_mean_std(df_fold_sum["All Damages - Settlement Zone (> Thr)"])
    dist_m, _, dist_str = format_mean_std(df_fold_sum["All Damages - Undamaged Zone (<= Thr)"])
    glob_m, glob_s, glob_str = format_mean_std(df_fold_sum["Global Accuracy"])

    best_intact = float(best_fold_row["Intact (test)"])
    best_d1 = float(best_fold_row["Damage 1 (Overall)"])
    best_d2 = float(best_fold_row["Damage 2 (Overall)"])
    best_d3 = float(best_fold_row["Damage 3 (Overall)"])
    best_d4 = float(best_fold_row["Damage 4 (Overall)"])
    best_d5 = float(best_fold_row["Damage 5 (Overall)"])
    best_kop = float(best_fold_row["All Damages - Settlement Zone (> Thr)"])
    best_dist = float(best_fold_row["All Damages - Undamaged Zone (<= Thr)"])
    best_glob = float(best_fold_row["Global Accuracy"])

    summary_row = {
        "Model": model_label,
        "Weight": weight_label,
        "Best_Fold (Lowest Loss)": f"Fold {best_fold_idx}",
        "Best_Fold_Total_Loss": round(float(best_fold_row["Total_Loss"]), 5),
        "Total_Loss (4-Fold Mean ± Std)": tot_str,
        "Intact (test) (4-Fold Mean ± Std %)": intact_str,
        "Intact (test) [Best Fold] (%)": round(best_intact, 2),
        "Damage 1 [g6 > Thr] (4-Fold %)": d1_str,
        "Damage 1 [Best Fold] (%)": round(best_d1, 2),
        "Damage 2 [g5-g6 > Thr] (4-Fold %)": d2_str,
        "Damage 2 [Best Fold] (%)": round(best_d2, 2),
        "Damage 3 [g4-g6 > Thr] (4-Fold %)": d3_str,
        "Damage 3 [Best Fold] (%)": round(best_d3, 2),
        "Damage 4 [g4-g6 > Thr] (4-Fold %)": d4_str,
        "Damage 4 [Best Fold] (%)": round(best_d4, 2),
        "Damage 5 [g4-g6 > Thr] (4-Fold %)": d5_str,
        "Damage 5 [Best Fold] (%)": round(best_d5, 2),
        "Damaged Zone Avg (4-Fold %)": kop_str,
        "Damaged Zone Avg [Best Fold] (%)": round(best_kop, 2),
        "Undamaged Zone Avg (4-Fold %)": dist_str,
        "Undamaged Zone Avg [Best Fold] (%)": round(best_dist, 2),
        "Global Accuracy (4-Fold Mean ± Std %)": glob_str,
        "Best_Fold_Global_Accuracy (%)": round(best_glob, 2),
        "_global_mean_numeric": glob_m,
        "_global_std_numeric": glob_s,
        "_best_fold_global_acc_numeric": best_glob,
        "_best_fold_loss_numeric": float(best_fold_row["Total_Loss"]),
        "_best_fold_idx": best_fold_idx,
    }

    return df_all_folds_groups, df_group_agg, summary_row, loss_row, best_fold_idx


# ==============================================================================
# 7. STATISTICAL HYPOTHESIS TESTING (ONE-SIDED MANN-WHITNEY U TEST)
# ==============================================================================
def p_value_to_stars(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    return "ns"


def perform_hypothesis_tests_vs_intact_train(
    model_label, weight_label, fold_dfs_dict, best_fold_idx
):
    """
    Tests whether the boxplot distributions of DI-KDE for 'Intact (test)' and
    'Damage 1'..'Damage 5' are statistically greater than 'Intact (train)' in each group (g1..g6).
    Evaluates both across the 4 folds and specifically on the BEST FOLD (plotted in the boxplot).
    """
    hypothesis_rows = []
    df_best = fold_dfs_dict[best_fold_idx]

    for grp_name in GROUP_ORDER:
        best_train = df_best[
            (df_best["Group"] == grp_name) & (df_best["Scenario"] == "Intact (train)")
        ]["DI"].values

        if len(best_train) == 0:
            continue

        for test_scen in TEST_SCENARIO_LABELS:
            fold_p_vals = []
            fold_cliffs = []
            fold_cles = []
            fold_med_ratios = []
            folds_rejected = 0

            for f_idx, df_f in sorted(fold_dfs_dict.items()):
                f_train = df_f[(df_f["Group"] == grp_name) & (df_f["Scenario"] == "Intact (train)")]["DI"].values
                f_scen = df_f[(df_f["Group"] == grp_name) & (df_f["Scenario"] == test_scen)]["DI"].values
                if len(f_train) > 0 and len(f_scen) > 0:
                    u_stat, p_val = mannwhitneyu(f_scen, f_train, alternative="greater")
                    cles = (u_stat / (len(f_scen) * len(f_train))) * 100.0
                    cliffs_d = (2.0 * u_stat) / (len(f_scen) * len(f_train)) - 1.0
                    med_ratio = float(np.median(f_scen) / np.median(f_train)) if np.median(f_train) > 0 else np.nan

                    fold_p_vals.append(p_val)
                    fold_cles.append(cles)
                    fold_cliffs.append(cliffs_d)
                    fold_med_ratios.append(med_ratio)
                    if p_val < ALPHA_SIGNIFICANCE:
                        folds_rejected += 1

            best_scen = df_best[
                (df_best["Group"] == grp_name) & (df_best["Scenario"] == test_scen)
            ]["DI"].values
            if len(best_scen) == 0:
                continue

            best_u, best_p = mannwhitneyu(best_scen, best_train, alternative="greater")
            best_cles = (best_u / (len(best_scen) * len(best_train))) * 100.0
            best_cliffs = (2.0 * best_u) / (len(best_scen) * len(best_train)) - 1.0
            stars = p_value_to_stars(best_p)

            _, _, cles_str = format_mean_std(fold_cles, decimals=2)
            _, _, cliffs_str = format_mean_std(fold_cliffs, decimals=3)
            _, _, ratio_str = format_mean_std(fold_med_ratios, decimals=2)

            expected_pos = EXPECTED_LOCALIZATION_STATE[test_scen][grp_name]
            expected_hyp = (
                "Reject H0 (Damaged: DI > Train)"
                if expected_pos == "Above"
                else "Undamaged Zone (Expected <= Threshold)"
            )

            hypothesis_rows.append({
                "Model": model_label,
                "Weight": weight_label,
                "Best_Fold": f"Fold {best_fold_idx}",
                "Scenario (vs. Intact train)": test_scen,
                "Sensor_Group": grp_name,
                "Expected_Physical_State": expected_hyp,
                "Folds_Significant (p < 0.05)": f"{folds_rejected}/{len(fold_p_vals)}",
                "Median_Ratio_4Folds (Mean ± Std)": ratio_str,
                "CLES_4Folds (Mean ± Std %)": cles_str,
                "Cliffs_Delta_4Folds (Mean ± Std)": cliffs_str,
                "Best_Fold_MannWhitney_U": round(float(best_u), 1),
                "Best_Fold_CLES (%)": round(float(best_cles), 2),
                "Best_Fold_Cliffs_Delta": round(float(best_cliffs), 3),
                "Best_Fold_p_value (one-sided)": float(f"{best_p:.4e}"),
                "Significance": stars,
                "Reject_H0 (Best Fold DI > Train)": "Yes" if best_p < ALPHA_SIGNIFICANCE else "No",
            })

    return pd.DataFrame(hypothesis_rows)


# ==============================================================================
# 8. CLEAN BOXPLOT GENERATION (BEST FOLD OF BEST WEIGHT ONLY)
# ==============================================================================
PALETTE_SCENARIOS = {
    "Intact (train)": "#FBBC05",
    "Intact (test)": "#34A853",
    "Damage 1": "#4285F4",
    "Damage 2": "#AA46BB",
    "Damage 3": "#FF7043",
    "Damage 4": "#EA4335",
    "Damage 5": "#8D6E63",
}


def plot_single_model_boxplot(df_best_fold, group_thresholds, display_title, file_tag, save_dir):
    """Generates the Z24 Bridge boxplot for the best fold (lowest loss) of a model family."""
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.boxplot(
        data=df_best_fold,
        x="Group",
        y="DI",
        hue="Scenario",
        ax=ax,
        palette=PALETTE_SCENARIOS,
        showfliers=False,
    )

    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(14)

    for i, grp_name in enumerate(GROUP_ORDER):
        if grp_name in group_thresholds:
            ax.hlines(
                y=group_thresholds[grp_name],
                xmin=i - 0.4,
                xmax=i + 0.4,
                colors="purple",
                linestyles="--",
                label=f"Threshold ({THRESHOLD_PERCENTILE}%)" if i == 0 else "",
            )

    ax.set_title(f"Z24 Bridge - {display_title}", fontsize=16, fontname="Times New Roman", fontweight="bold")
    ax.set_xlabel("Group of sensors", fontsize=14, fontname="Times New Roman")
    ax.set_ylabel("Global damage index (DI-KDE) [log-scale]", fontsize=14, fontname="Times New Roman")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.6)

    legend = plt.legend(
        title="Damage scenario",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        prop={"family": "Times New Roman", "size": 12},
    )
    plt.setp(legend.get_title(), fontname="Times New Roman", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.86, 1])

    out_path = os.path.join(save_dir, f"boxplot_z24_best_fold_{file_tag}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved best-fold boxplot: {out_path}")


def plot_combined_best_models_boxplots(best_runs_map, save_dir):
    """Generates a multi-panel figure comparing the best fold (lowest loss) of the best weight of each model."""
    model_keys = list(best_runs_map.keys())
    n_models = len(model_keys)
    if n_models == 0:
        return

    ncols = 2
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 6.5 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for idx, model_label in enumerate(model_keys):
        ax = axes[idx]
        run_info = best_runs_map[model_label]
        df_model = run_info["df_best_fold"]
        group_thresholds = run_info["thresholds_best_fold"]
        display_title = run_info["display_title"]

        sns.boxplot(
            data=df_model,
            x="Group",
            y="DI",
            hue="Scenario",
            ax=ax,
            palette=PALETTE_SCENARIOS,
            showfliers=False,
        )

        for i, grp_name in enumerate(GROUP_ORDER):
            if grp_name in group_thresholds:
                ax.hlines(
                    y=group_thresholds[grp_name],
                    xmin=i - 0.4,
                    xmax=i + 0.4,
                    colors="purple",
                    linestyles="--",
                    label=f"Threshold ({THRESHOLD_PERCENTILE}%)" if i == 0 else "",
                )

        panel_letter = chr(ord("a") + idx)
        ax.set_title(f"({panel_letter}) {display_title}", fontsize=15, fontname="Times New Roman", fontweight="bold")
        ax.set_xlabel("Group of sensors", fontsize=13, fontname="Times New Roman")
        ax.set_ylabel("Global damage index (DI-KDE) [log-scale]", fontsize=13, fontname="Times New Roman")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.6)

        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.legend(
        handles,
        labels,
        title="Damage scenario",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        prop={"family": "Times New Roman", "size": 12},
    )
    plt.tight_layout()
    out_path = os.path.join(save_dir, "boxplot_z24_best_folds_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined comparison boxplot (best fold per model): {out_path}")


# ==============================================================================
# 9. EXPORT TABLES (4-FOLD MEAN ± STD + BEST FOLD ACCURACY + LOSS + HYPOTHESIS)
# ==============================================================================
def export_all_tables(
    all_raw_fold_dfs,
    all_group_agg_dfs,
    all_summary_rows,
    all_loss_rows,
    all_hypothesis_dfs,
    best_runs_map,
    save_dir,
):
    df_raw_folds = pd.concat(all_raw_fold_dfs, ignore_index=True)
    df_group_agg = pd.concat(all_group_agg_dfs, ignore_index=True)
    df_summary = pd.DataFrame(all_summary_rows)
    df_losses = pd.DataFrame(all_loss_rows)
    df_hypothesis_all = pd.concat(all_hypothesis_dfs, ignore_index=True)

    df_summary["Selected_For_Boxplot"] = df_summary.apply(
        lambda row: "Yes (*)"
        if best_runs_map.get(row["Model"], {}).get("weight_label") == row["Weight"]
        else "No",
        axis=1,
    )
    df_losses["Selected_For_Boxplot"] = df_summary["Selected_For_Boxplot"]

    df_summary_clean = df_summary.drop(
        columns=[
            "_global_mean_numeric",
            "_global_std_numeric",
            "_best_fold_global_acc_numeric",
            "_best_fold_loss_numeric",
            "_best_fold_idx",
        ],
        errors="ignore",
    )

    df_group_agg["Model_Weight"] = df_group_agg.apply(
        lambda r: f"{r['Model']} ({r['Weight']})" if r["Weight"] not in ["N/A", "0.1 (λ_class)"] else r["Model"],
        axis=1,
    )
    col_order = []
    for s in all_summary_rows:
        mw = f"{s['Model']} ({s['Weight']})" if s["Weight"] not in ["N/A", "0.1 (λ_class)"] else s["Model"]
        if mw not in col_order:
            col_order.append(mw)

    # Pivot 1: 4-Fold Mean ± Std + Best Fold Accuracy per group
    df_pivot_groups_combined = df_group_agg.pivot(
        index=["Scenario", "Sensor_Group", "Expected_Position"],
        columns="Model_Weight",
        values="Combined_4Fold_and_BestFold (%)",
    )[col_order].reset_index()

    # Pivot 2: Best Fold Accuracy (%) only per group
    df_pivot_groups_best_fold = df_group_agg.pivot(
        index=["Scenario", "Sensor_Group", "Expected_Position"],
        columns="Model_Weight",
        values="Best_Fold_Accuracy (%)",
    )[col_order].reset_index()

    best_hyp_dfs = [info["df_hyp"] for info in best_runs_map.values() if info.get("df_hyp") is not None]
    df_hypothesis_best = pd.concat(best_hyp_dfs, ignore_index=True) if best_hyp_dfs else df_hypothesis_all

    print("\n" + "=" * 150)
    print("1) Z24 BRIDGE LOSS DECOMPOSITION: RECONSTRUCTION (SSIM), PENALTY (RAW & WEIGHTED), TOTAL LOSS, AND BEST FOLD")
    print("=" * 150)
    print(df_losses.to_string(index=False))

    print("\n" + "=" * 150)
    print("2) Z24 BRIDGE LOCALIZATION ACCURACY: 4-FOLD (MEAN ± STD %) AND BEST FOLD ACCURACY (%) BY SCENARIO & GLOBAL")
    print("=" * 150)
    print(df_summary_clean.to_string(index=False))

    print("\n" + "=" * 150)
    print("3) Z24 BRIDGE GROUP-LEVEL (g1..g6) ACCURACY: 4-FOLD (MEAN ± STD %) + BEST FOLD (%)")
    print("=" * 150)
    print(df_pivot_groups_combined.to_string(index=False))

    print("\n" + "=" * 150)
    print("4) ONE-SIDED MANN-WHITNEY U HYPOTHESIS TEST (H1: DI_SCENARIO > DI_INTACT_TRAIN) FOR BEST WEIGHTS")
    print("=" * 150)
    print(df_hypothesis_best.to_string(index=False))
    print("=" * 150)

    losses_csv = os.path.join(save_dir, "z24_loss_decomposition_4folds_and_best_fold.csv")
    summary_csv = os.path.join(save_dir, "z24_accuracy_summary_4folds_and_best_fold.csv")
    pivot_comb_csv = os.path.join(save_dir, "z24_group_pivot_4folds_and_best_fold.csv")
    pivot_best_csv = os.path.join(save_dir, "z24_group_pivot_best_fold_only.csv")
    hyp_best_csv = os.path.join(save_dir, "z24_hypothesis_tests_best_weights.csv")
    hyp_all_csv = os.path.join(save_dir, "z24_hypothesis_tests_all_weights.csv")
    excel_path = os.path.join(save_dir, "z24_complete_statistical_analysis.xlsx")

    df_losses.to_csv(losses_csv, index=False)
    df_summary_clean.to_csv(summary_csv, index=False)
    df_pivot_groups_combined.to_csv(pivot_comb_csv, index=False)
    df_pivot_groups_best_fold.to_csv(pivot_best_csv, index=False)
    df_hypothesis_best.to_csv(hyp_best_csv, index=False)
    df_hypothesis_all.to_csv(hyp_all_csv, index=False)

    with pd.ExcelWriter(excel_path) as writer:
        df_losses.to_excel(writer, sheet_name="1_Loss_Decomposition", index=False)
        df_summary_clean.to_excel(writer, sheet_name="2_Accuracy_4Fold_and_BestFold", index=False)
        df_pivot_groups_combined.to_excel(writer, sheet_name="3_Group_Pivot_Combined", index=False)
        df_pivot_groups_best_fold.to_excel(writer, sheet_name="4_Group_Pivot_BestFold_Only", index=False)
        df_hypothesis_best.to_excel(writer, sheet_name="5_Hypothesis_Tests_Best", index=False)
        df_hypothesis_all.to_excel(writer, sheet_name="6_Hypothesis_Tests_All", index=False)
        df_raw_folds.to_excel(writer, sheet_name="7_Raw_Per_Fold_Metrics", index=False)

    print(
        f"\nAll Z24 Bridge tables exported successfully to:\n"
        f"  - {losses_csv}\n"
        f"  - {summary_csv}\n"
        f"  - {pivot_comb_csv}\n"
        f"  - {pivot_best_csv}\n"
        f"  - {hyp_best_csv}\n"
        f"  - {excel_path}"
    )


# ==============================================================================
# 10. MAIN EXECUTION
# ==============================================================================
def main():
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")
    print(f"BASE_DIR_OUTPUT : {BASE_DIR_OUTPUT}")
    print(f"ROOT_DATA_DIR   : {ROOT_DATA_DIR}")
    print(f"RESULTS_SAVE_DIR: {RESULTS_SAVE_DIR}")

    sensors_list, sensor_groups_map = resolve_sensor_names_and_groups(
        ROOT_DATA_DIR, INTACT_CONDITION_NAME
    )
    print(f"\nDetected Z24 Sensors ({len(sensors_list)}): {sensors_list}")
    print(f"Overlapping Sensor Groups (g1..g6): {sensor_groups_map}")
    print(f"Damaged Groups by Settlement Scenario: {DAMAGED_GROUPS_FOR_SETTLEMENT}")

    print("\n--- Loading Z24 Bridge CWT Dataset across all 7 sensors and 7 scenarios ---")
    all_sensor_data = load_all_sensor_data(
        ROOT_DATA_DIR, DAMAGE_SCENARIOS, sensors_list
    )

    all_raw_fold_dfs = []
    all_group_agg_dfs = []
    all_summary_rows = []
    all_loss_rows = []
    all_hypothesis_dfs = []
    best_runs_map = {}

    for model_label, cfg in MODEL_CONFIGS.items():
        print(f"\n" + "=" * 100)
        print(f"Evaluating Model Family on Z24 Bridge: {model_label}")
        print("=" * 100)

        variations = find_all_weight_models_from_config(model_label, cfg)
        if not variations:
            print(f"Skipping {model_label} (no valid Optuna checkpoints found).")
            continue

        family_evaluated_runs = []

        for var in variations:
            w_label = var["weight_label"]
            w_val = var["weight_value"]
            val_loss = var["val_loss"]
            trial_num = var["trial_number"]
            fold_ckpts = var["fold_ckpts"]
            best_params = var["best_params"]

            print(
                f"\n-> {model_label} | Weight: {w_label} | Best Trial #{trial_num} "
                f"(Optuna CV Loss: {val_loss:.5f}) | Folds found: {list(fold_ckpts.keys())}"
            )

            fold_dfs_dict = {}
            fold_thresholds_dict = {}
            fold_group_acc_dfs = []
            fold_summaries = []

            for fold_idx, ckpt_path in sorted(fold_ckpts.items()):
                try:
                    model = instantiate_and_load_model(cfg, ckpt_path, best_params, device)

                    # 1. Evaluate Loss Decomposition (Recon Loss, Penalty Raw, Penalty Weighted, Total Loss)
                    loss_dict = evaluate_fold_loss_decomposition(
                        model, cfg["type"], w_val, best_params, all_sensor_data, sensors_list, device
                    )

                    # 2. Evaluate Hierarchical DI-KDE and Group Thresholds
                    df_fold, thr_fold = evaluate_single_fold_hierarchy(
                        model_label,
                        w_label,
                        fold_idx,
                        model,
                        cfg["type"],
                        all_sensor_data,
                        sensors_list,
                        sensor_groups_map,
                        device,
                    )
                    if df_fold.empty:
                        continue

                    fold_dfs_dict[fold_idx] = df_fold
                    fold_thresholds_dict[fold_idx] = thr_fold

                    # 3. Compute Fold Localization Accuracies
                    df_groups_fold, fold_sum = compute_single_fold_accuracy(
                        model_label, w_label, fold_idx, df_fold, thr_fold, loss_dict
                    )
                    if not df_groups_fold.empty:
                        fold_group_acc_dfs.append(df_groups_fold)
                        fold_summaries.append(fold_sum)
                        print(
                            f"   [Fold {fold_idx}] Total Loss: {loss_dict['total_loss']:.5f} "
                            f"(Recon: {loss_dict['recon_loss']:.5f}, Pen_W: {loss_dict['penalty_weighted']:.5f}) | "
                            f"Intact (test): {fold_sum['Intact (test)']:.2f}% | "
                            f"Damaged Zone: {fold_sum['All Damages - Settlement Zone (> Thr)']:.2f}% | "
                            f"Global Acc: {fold_sum['Global Accuracy']:.2f}%"
                        )
                except Exception as e:
                    print(f"   Error in {model_label} (Weight={w_label}, Fold={fold_idx}): {e}")

            if not fold_summaries:
                continue

            # Summarize 4-fold Mean ± Std AND identify the Best Fold (lowest Total_Loss)
            (
                df_raw_folds,
                df_group_agg,
                summary_row,
                loss_row,
                best_fold_idx,
            ) = summarize_4fold_and_best_fold_metrics(
                model_label, w_label, val_loss, fold_group_acc_dfs, fold_summaries
            )
            all_raw_fold_dfs.append(df_raw_folds)
            all_group_agg_dfs.append(df_group_agg)
            all_summary_rows.append(summary_row)
            all_loss_rows.append(loss_row)

            # Perform one-sided Mann-Whitney U Hypothesis Test (DI_scenario > DI_Intact_train)
            df_hyp = perform_hypothesis_tests_vs_intact_train(
                model_label, w_label, fold_dfs_dict, best_fold_idx
            )
            if not df_hyp.empty:
                all_hypothesis_dfs.append(df_hyp)

            # Extract the Best Fold DataFrame and Thresholds for Boxplot generation
            df_best_fold = fold_dfs_dict[best_fold_idx].copy()
            thr_best_fold = fold_thresholds_dict[best_fold_idx].copy()

            display_title = (
                f"{model_label} (Weight = {w_label}, Best Fold = {best_fold_idx})"
                if cfg.get("multi_weight", False)
                else f"{model_label} (Best Fold = {best_fold_idx})"
            )

            family_evaluated_runs.append({
                "model_label": model_label,
                "weight_label": w_label,
                "val_loss": val_loss,
                "best_fold_idx": best_fold_idx,
                "best_fold_loss": summary_row["_best_fold_loss_numeric"],
                "best_fold_global_acc": summary_row["_best_fold_global_acc_numeric"],
                "global_acc_mean": summary_row["_global_mean_numeric"],
                "global_acc_std": summary_row["_global_std_numeric"],
                "global_acc_str": summary_row["Global Accuracy (4-Fold Mean ± Std %)"],
                "df_best_fold": df_best_fold,
                "thresholds_best_fold": thr_best_fold,
                "df_hyp": df_hyp,
                "display_title": display_title,
            })

            print(
                f"   ==> [4-Fold Mean ± Std] Global Acc: {summary_row['Global Accuracy (4-Fold Mean ± Std %)']}% | "
                f"[Best Fold #{best_fold_idx}] Loss: {summary_row['Best_Fold_Total_Loss']:.5f} | "
                f"Best Fold Global Acc: {summary_row['Best_Fold_Global_Accuracy (%)']:.2f}%"
            )

        # Select the best weight configuration for this model family to generate its Boxplot
        if family_evaluated_runs:
            if BEST_WEIGHT_SELECTION_METRIC == "global_accuracy":
                family_evaluated_runs.sort(
                    key=lambda r: (
                        r["global_acc_mean"],
                        r["best_fold_global_acc"],
                        -r["best_fold_loss"],
                    ),
                    reverse=True,
                )
            elif BEST_WEIGHT_SELECTION_METRIC == "best_fold_loss":
                family_evaluated_runs.sort(key=lambda r: r["best_fold_loss"])
            else:
                family_evaluated_runs.sort(key=lambda r: r["val_loss"])

            best_run = family_evaluated_runs[0]
            best_runs_map[model_label] = best_run
            print(
                f"\n★ Selected for {model_label} Boxplot: Weight = {best_run['weight_label']} | "
                f"Best Fold = Fold {best_run['best_fold_idx']} (Total Loss = {best_run['best_fold_loss']:.5f}, "
                f"Best Fold Acc = {best_run['best_fold_global_acc']:.2f}%, "
                f"4-Fold Acc = {best_run['global_acc_str']}%)"
            )

            file_tag = model_label.lower().replace(" ", "_").replace("-", "_")
            plot_single_model_boxplot(
                best_run["df_best_fold"],
                best_run["thresholds_best_fold"],
                best_run["display_title"],
                file_tag,
                RESULTS_SAVE_DIR,
            )

    # Export combined boxplot (best fold per model) and all statistical/sensitivity/loss tables
    if best_runs_map:
        plot_combined_best_models_boxplots(best_runs_map, RESULTS_SAVE_DIR)
        export_all_tables(
            all_raw_fold_dfs,
            all_group_agg_dfs,
            all_summary_rows,
            all_loss_rows,
            all_hypothesis_dfs,
            best_runs_map,
            RESULTS_SAVE_DIR,
        )
    else:
        print("\nNo models were successfully evaluated. Please verify your directory paths.")


if __name__ == "__main__":
    main()