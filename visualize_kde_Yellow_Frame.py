import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import optuna
from itertools import combinations
from datetime import datetime
# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================

# --- Optuna Study Directory Settings ---
BASE_DIR_OUTPUT = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo"
OUTPUT_DIR = os.path.join(BASE_DIR_OUTPUT, 'optuna_optimization_output_yellow_frame')
STUDY_NAME = "ccae_hyperparam_optimization_yellow_frame"
DB_FILENAME = "ccae_optimization_yellow_frame.db"

# --- Analysis Data Directory Settings ---
ROOT_DATA_DIR = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo/imagens_cwt_yellow_frame"

# --- Fixed Model Settings (IDENTICAL TO TRAINING) ---
FIXED_PARAMS = {
    'classifier_linear_dim': 64,
    'bottleneck_channels': 4
}

# --- Configurações do Dataset e Análise ---
ALL_SENSORS_MODEL_TRAINED_ON = [f'Sensor{i}' for i in range(1, 16)]
NUM_SENSORS_IN_TRAINED_MODEL = len(ALL_SENSORS_MODEL_TRAINED_ON)
DAMAGE_SCENARIOS = ['d_0_intact', 'd_0_unknown', 'd_1', 'd_2']
INTACT_CONDITION_NAME = 'd_0_intact'

# ==============================================================================
# <<<<<<<<<<<<<<<<<<< DISPLAY CONFIGURATION >>>>>>>>>>>>>>>>>
# ==============================================================================
# Choose which 'floor' (sensor group) you want to view.

# Options: '1st_floor', '2nd_floor', '3rd_floor', '4th_floor', '5th_floor'
FLOOR_TO_VISUALIZE = '3_andar'
# ==============================================================================

# ==============================================================================
# 2. DEFINITION OF THE CCAE MODEL AND AUXILIARY FUNCTIONS
# ==============================================================================

class CCAE(nn.Module):
    def __init__(self, num_sensors, bottleneck_channels, classifier_linear_dim, dropout_rate):
        super(CCAE, self).__init__()
        self.num_sensors = num_sensors
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_channels), nn.ReLU()
        )
        self.sensor_embedding = nn.Embedding(num_sensors, 64)
        classifier_input_dim = bottleneck_channels * 8 * 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_input_dim, classifier_linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_linear_dim, num_sensors)
        )
        decoder_input_channels = bottleneck_channels + 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(decoder_input_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )

    def get_latent_features(self, x, scaling_factor):
        with torch.no_grad():
            x_encoded = self.encoder(x)
            encoded_features = self.bottleneck_conv(x_encoded)
            sensor_logits = self.classifier(encoded_features)
            sensor_probabilities = torch.softmax(sensor_logits, dim=1)
            all_sensor_embeddings = self.sensor_embedding(torch.arange(self.num_sensors).to(x.device))
            combined_sensor_emb = (sensor_probabilities.unsqueeze(2) * all_sensor_embeddings.unsqueeze(0)).sum(dim=1)
            combined_sensor_emb_reshaped = combined_sensor_emb.view(-1, 1, 8, 8) * scaling_factor
            concatenated_features = torch.cat([encoded_features, combined_sensor_emb_reshaped], dim=1)
            pooled_latent = nn.AdaptiveAvgPool2d((1, 1))(concatenated_features)
            return pooled_latent.view(pooled_latent.size(0), -1)


def find_best_model_from_study(study_name, storage_url, output_dir):
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        best_trial = study.best_trial
        print(f"--- Melhor Trial Encontrado (Nº {best_trial.number}) ---")
        model_path = os.path.join(output_dir, f'trial_{best_trial.number}', 'best_model_fold_1.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado em: {model_path}")
        return model_path, best_trial.params
    except Exception as e:
        print(f"❌ Erro ao carregar o estudo Optuna: {e}")
        return None, None

class CWTDatasetAnalysis(Dataset):
    def __init__(self, root_folder):
        self.image_paths = glob.glob(os.path.join(root_folder, '**', '*.png'), recursive=True)
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
            return None, None
            
def collate_fn_analysis(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch: return torch.tensor([]), []
    images, names = zip(*batch)
    return torch.stack(images), list(names)

def load_all_sensor_data(root_dir, damage_scenarios, sensors_for_analysis):
    all_data = {sensor: {} for sensor in sensors_for_analysis}
    for sensor_name in sensors_for_analysis:
        for condition in damage_scenarios:
            path = os.path.join(root_dir, condition, sensor_name)
            if os.path.isdir(path):
                dataset = CWTDatasetAnalysis(path)
                if len(dataset) > 0:
                    all_data[sensor_name][condition] = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_analysis)
    return all_data

def extract_latent_features(model, dataloader, device, scaling_factor):
    model.eval()
    features_dict = {}
    with torch.no_grad():
        for images, names in dataloader:
            if images.numel() == 0: continue
            images = images.to(device)
            latent_features = model.get_latent_features(images, scaling_factor).cpu().numpy()
            for i, name in enumerate(names):
                features_dict[name] = latent_features[i]
    return features_dict

def calculate_mahalanobis_distance(features, mean, inv_cov):
    if not features: return np.array([])
    features_array = np.array(features)
    if features_array.ndim == 1: features_array = features_array.reshape(1, -1)
    distances = [mahalanobis(f, mean, inv_cov) for f in features_array]
    return np.array(distances)

def load_best_model(model_path, device, num_sensors, fixed_params, best_hyperparams):
    model = CCAE(
        num_sensors=num_sensors,
        bottleneck_channels=fixed_params['bottleneck_channels'],
        classifier_linear_dim=fixed_params['classifier_linear_dim'],
        dropout_rate=best_hyperparams['dropout_rate']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def calculate_scaling_factor_all_sensors(model, dataloaders, device):
    model.eval()
    encoded_norms, sensor_emb_norms = [], []
    with torch.no_grad():
        for dataloader in dataloaders:
            for images, _ in dataloader:
                if images.numel() == 0: continue
                images = images.to(device)
                x_encoded = model.encoder(images)
                encoded_features = model.bottleneck_conv(x_encoded)
                sensor_logits = model.classifier(encoded_features)
                sensor_probabilities = torch.softmax(sensor_logits, dim=1)
                all_sensor_embeddings = model.sensor_embedding(torch.arange(model.num_sensors).to(device))
                combined_sensor_emb = (sensor_probabilities.unsqueeze(2) * all_sensor_embeddings.unsqueeze(0)).sum(dim=1)
                combined_sensor_emb_reshaped = combined_sensor_emb.view(-1, 1, 8, 8)
                encoded_norms.append(torch.linalg.norm(encoded_features.flatten(start_dim=1), dim=1).cpu().numpy())
                sensor_emb_norms.append(torch.linalg.norm(combined_sensor_emb_reshaped.flatten(start_dim=1), dim=1).cpu().numpy())
    if not encoded_norms: return 1.0
    avg_encoded_norm = np.mean(np.concatenate(encoded_norms))
    avg_sensor_emb_norm = np.mean(np.concatenate(sensor_emb_norms))
    return avg_encoded_norm / avg_sensor_emb_norm if avg_sensor_emb_norm > 0 else 1.0


# ==============================================================================
# 3. VISUALIZAÇÃO DO KDE
# ==============================================================================

def plot_kde_visualizations(kde_model, intact_mds_array, damaged_mds_dict, sensor_names, floor_name):
    """
    Generates and saves a grid of 2D contour plots to visualize KDE density.

    Args:
        kde_model (KernelDensity): The trained KDE model.
        intact_mds_array (np.array): Array of MD vectors of the intact condition (N_samples, N_sensors).
        damaged_mds_dict (dict): Dictionary with MD vectors of damage scenarios.
Ex: {'d_1': array, 'd_2': array}.
        sensor_names (list): Lista de nomes de sensores para os eixos do plot.
        floor_name (str): Floor/group name for the plot title.
    """
    num_sensors = intact_mds_array.shape[1]
    if num_sensors < 2:
        print("❌ Visualização requer pelo menos 2 sensores no grupo.")
        return

    # Generates all combinations of sensor pairs
    sensor_pairs = list(combinations(range(num_sensors), 2))
    
    # Define the plot grid layout
    n_cols = 3
    n_rows = (len(sensor_pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    # Calculates the average of intact MDs to use in unplotted dimensions.
    mean_mds_intact = np.mean(intact_mds_array, axis=0)
    
    print(f"\n--- Gerando visualizações para {floor_name} ---")

    for i, (dim1, dim2) in enumerate(sensor_pairs):
        ax = axes[i]

        # Define the 2D grid for KDE evaluation
        x_min, x_max = intact_mds_array[:, dim1].min(), intact_mds_array[:, dim1].max()
        y_min, y_max = intact_mds_array[:, dim2].min(), intact_mds_array[:, dim2].max()
        x_pad = (x_max - x_min) * 0.2
        y_pad = (y_max - y_min) * 0.2
        xx, yy = np.mgrid[x_min - x_pad : x_max + x_pad : 100j, 
                          y_min - y_pad : y_max + y_pad : 100j]

        # Creates N-dimensional evaluation points
        eval_points = np.tile(mean_mds_intact, (xx.ravel().shape[0], 1))
        eval_points[:, dim1] = xx.ravel()
        eval_points[:, dim2] = yy.ravel()

        # Evaluate KDE on the grid and calculate density
        log_density = kde_model.score_samples(eval_points)
        Z = np.exp(log_density).reshape(xx.shape)

        # Plots the density contour (heatmap)
        contour = ax.contourf(xx, yy, Z, levels=15, cmap='viridis_r')
        if i >= 2:  
            fig.colorbar(contour, ax=ax, label='Probability Density Function $p_g(.)$')

        # Overlays data points
        # 1. Intact data (base for KDE)
        ax.scatter(intact_mds_array[:, dim1], intact_mds_array[:, dim2], 
                   s=10, c='white', edgecolor='black', alpha=0.7, label='Intact (train)')
                   
        #2. Damage Data
        colors = {'d_1': 'red', 'd_2': 'orange'}
        scenario_labels = {
            'd_0_unknown': 'Intact (test)',
            'd_1': 'Damage 1',
            'd_2': 'Damage 2'
        }
        for scenario, mds_array in damaged_mds_dict.items():
            if mds_array.size > 0:
                label_text = scenario_labels.get(scenario, scenario)
                ax.scatter(mds_array[:, dim1], mds_array[:, dim2], 
                           s=20, c=colors.get(scenario, 'magenta'), marker='X', label=label_text)

        ax.set_xlabel(f"MD {sensor_names[dim1].replace('Sensor', 'Sensor ')}")
        ax.set_ylabel(f"MD {sensor_names[dim2].replace('Sensor', 'Sensor ')}")
        subplot_titles = ['(a)', '(b)', '(c)']
        ax.set_title(subplot_titles[i] if i < 3 else f'{sensor_names[dim1]} vs {sensor_names[dim2]}')
        if i == 0:  # The legend is only displayed on the first graph.
            ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    #fig.suptitle(f'Visualização da Densidade do KDE para o Grupo "{floor_name}"', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plots_save_dir = os.path.join(os.path.dirname(ROOT_DATA_DIR), 'plots_kde_visualization')
    os.makedirs(plots_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"kde_density_{floor_name}_{timestamp}.png"
    plt.savefig(os.path.join(plots_save_dir, plot_filename), dpi=200, bbox_inches='tight')
    print(f"\n✅ Plot de visualização do KDE salvo em: {os.path.join(plots_save_dir, plot_filename)}")
    plt.show()


# ==============================================================================
#4. MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    TAMANHO_FONTE_BASE = 17
    plt.rcParams.update({
        # --- Font Settings ---
        'font.family': 'times new roman',                   
        'mathtext.fontset': 'stix',                   
        
        # --- Size Settings ---
        'font.size': TAMANHO_FONTE_BASE,
        'axes.titlesize': TAMANHO_FONTE_BASE + 2,
        'axes.labelsize': TAMANHO_FONTE_BASE,
        'xtick.labelsize': TAMANHO_FONTE_BASE - 2,
        'ytick.labelsize': TAMANHO_FONTE_BASE - 2,
        'legend.fontsize': TAMANHO_FONTE_BASE - 2,
        'figure.titlesize': TAMANHO_FONTE_BASE + 4,
        'figure.titleweight': 'bold',
        'axes.titleweight': 'bold',
    })
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # --- STEP 1: Load the best trained model ---
    db_path = os.path.join(OUTPUT_DIR, DB_FILENAME)
    storage_url = f"sqlite:///{db_path}"
    best_model_path, best_hyperparams = find_best_model_from_study(STUDY_NAME, storage_url, OUTPUT_DIR)
    
    if not best_model_path:
        exit()

    model = load_best_model(best_model_path, device, NUM_SENSORS_IN_TRAINED_MODEL, FIXED_PARAMS, best_hyperparams)
    if model is None:
        exit()

    # --- STEP 2: Process the data (same as the original script) ---
    sensor_groups_by_floor = {
        '1_andar': ['Sensor1', 'Sensor2', 'Sensor3'],
        '2_andar': ['Sensor4', 'Sensor5', 'Sensor6'],
        '3_andar': ['Sensor7', 'Sensor8', 'Sensor9'],
        '4_andar': ['Sensor10', 'Sensor11', 'Sensor12'],
        '5_andar': ['Sensor13', 'Sensor14', 'Sensor15']
    }
    
    if FLOOR_TO_VISUALIZE not in sensor_groups_by_floor:
        print(f"❌ Erro: '{FLOOR_TO_VISUALIZE}' não é um grupo de sensores válido.")
        exit()

    all_sensors_for_analysis = [s for floor_sensors in sensor_groups_by_floor.values() for s in floor_sensors]
    
    all_sensor_data = load_all_sensor_data(ROOT_DATA_DIR, DAMAGE_SCENARIOS, all_sensors_for_analysis)
    
    all_intact_dataloaders = [all_sensor_data[s][INTACT_CONDITION_NAME] for s in all_sensors_for_analysis if INTACT_CONDITION_NAME in all_sensor_data[s]]
    scaling_factor = calculate_scaling_factor_all_sensors(model, all_intact_dataloaders, device)
    print(f"Fator de Escala Global: {scaling_factor:.4f}")

    all_sensor_latent_features = {s: {c: {} for c in DAMAGE_SCENARIOS} for s in all_sensors_for_analysis}
    for sensor_name, conditions in all_sensor_data.items():
        for condition, dataloader in conditions.items():
            if dataloader:
                features = extract_latent_features(model, dataloader, device, scaling_factor)
                all_sensor_latent_features[sensor_name][condition].update(features)

    sensor_mahalanobis_indicators = {s: {c: {} for c in DAMAGE_SCENARIOS} for s in all_sensors_for_analysis}
    for sensor_name in all_sensors_for_analysis:
        intact_features = list(all_sensor_latent_features[sensor_name].get(INTACT_CONDITION_NAME, {}).values())
        if len(intact_features) > 1:
            try:
                mean_feature = np.mean(intact_features, axis=0)
                cov = EmpiricalCovariance().fit(intact_features).covariance_
                inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
                for condition in DAMAGE_SCENARIOS:
                    features_cond = list(all_sensor_latent_features[sensor_name].get(condition, {}).values())
                    filenames_cond = list(all_sensor_latent_features[sensor_name].get(condition, {}).keys())
                    if features_cond:
                        distances = calculate_mahalanobis_distance(features_cond, mean_feature, inv_cov)
                        sensor_mahalanobis_indicators[sensor_name][condition].update(zip(filenames_cond, distances))
            except np.linalg.LinAlgError:
                print(f"Aviso: Matriz singular para o sensor {sensor_name}. Pulando.")

    # --- STEP 3: Prepare data and train KDE for the selected floor. ---
    sensors_in_floor = sensor_groups_by_floor[FLOOR_TO_VISUALIZE]
    
    # Collect intact data for the floor.
    all_filenames_intact = set().union(*(sensor_mahalanobis_indicators[s].get(INTACT_CONDITION_NAME, {}).keys() for s in sensors_in_floor))
    floor_intact_mds_vectors = []
    for fname in sorted(list(all_filenames_intact)):
        md_vector = [sensor_mahalanobis_indicators[s][INTACT_CONDITION_NAME].get(fname) for s in sensors_in_floor]
        if all(v is not None for v in md_vector):
            floor_intact_mds_vectors.append(md_vector)
    intact_mds_array_floor = np.array(floor_intact_mds_vectors)

    if intact_mds_array_floor.shape[0] < 5:
        print(f"❌ Dados intactos insuficientes para treinar e visualizar o KDE para {FLOOR_TO_VISUALIZE}.")
        exit()

    # Collect damage data for the floor.
    damaged_mds_floor = {}
    for scenario in ['d_0_unknown','d_1', 'd_2']:
        all_filenames_damage = set().union(*(sensor_mahalanobis_indicators[s].get(scenario, {}).keys() for s in sensors_in_floor))
        floor_damage_mds_vectors = []
        for fname in sorted(list(all_filenames_damage)):
             md_vector = [sensor_mahalanobis_indicators[s][scenario].get(fname) for s in sensors_in_floor]
             if all(v is not None for v in md_vector):
                 floor_damage_mds_vectors.append(md_vector)
        damaged_mds_floor[scenario] = np.array(floor_damage_mds_vectors)

    # KDE Training
    kde_model_floor = KernelDensity(kernel='gaussian', bandwidth='scott')

    # Fit the model to the data
    kde_model_floor.fit(intact_mds_array_floor)

    # The optimal bandwidth is calculated internally and stored in the `bandwidth_` attribute
    print(f"Bandwidth selecionada pela regra de Scott: {kde_model_floor.bandwidth_:.4f}")

    # --- STEP 4: Visualization ---
    plot_kde_visualizations(
        kde_model=kde_model_floor,
        intact_mds_array=intact_mds_array_floor,
        damaged_mds_dict=damaged_mds_floor,
        sensor_names=sensors_in_floor,
        floor_name=FLOOR_TO_VISUALIZE

    )
