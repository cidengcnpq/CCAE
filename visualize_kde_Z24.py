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
# SEÇÃO DE CONFIGURAÇÃO
# ==============================================================================

# --- Configurações de Diretórios do Estudo Optuna ---
BASE_DIR_OUTPUT = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo"
OUTPUT_DIR = os.path.join(BASE_DIR_OUTPUT, 'optuna_optimization_output_Z24')
STUDY_NAME = "ccae_hyperparam_optimization_Z24"
DB_FILENAME = "ccae_optimization_Z24.db"

# --- Configurações de Diretórios dos Dados de Análise ---
ROOT_DATA_DIR = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo/imagens_CWT_Z24_v1_matlab"

# --- Configurações do Modelo Fixo (IDÊNTICO AO TREINAMENTO) ---
FIXED_PARAMS = {
    'classifier_linear_dim': 64,
    'bottleneck_channels': 4
}

# --- Configurações do Dataset e Análise ---
ALL_SENSORS_MODEL_TRAINED_ON = [f'Sensor{i}' for i in range(1, 8)]
NUM_SENSORS_IN_TRAINED_MODEL = len(ALL_SENSORS_MODEL_TRAINED_ON)
DAMAGE_SCENARIOS = ['d_0_intact', 'd_0_unknown', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5']
INTACT_CONDITION_NAME = 'd_0_intact'

# ==============================================================================
# <<<<<<<<<<<<<<<<<<<   CONFIGURAÇÃO DA VISUALIZAÇÃO   >>>>>>>>>>>>>>>>>
# ==============================================================================
# Grupos de sensores a visualizar
FLOORS_TO_VISUALIZE = ['g2', 'g5', 'g6']
# ==============================================================================


# NOTE: As classes e funções de `find_best_model_from_study` a `calculate_scaling_factor_all_sensors`
# são idênticas às do seu script original. Elas estão incluídas aqui para que este
# script seja executável de forma independente.

# ==============================================================================
# 2. DEFINIÇÃO DO MODELO CCAE E FUNÇÕES AUXILIARES (COPIADO DO SCRIPT ORIGINAL)
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

# Include find_best_model_from_study, CWTDatasetAnalysis, etc. (all helper functions from the original script)
# For brevity, these are assumed to be present and are omitted from this code block.
# Please copy them from your original script into this one.
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
    Gera e salva uma grade de plots de contorno 2D para visualizar a densidade do KDE.

    Args:
        kde_model (KernelDensity): O modelo KDE treinado.
        intact_mds_array (np.array): Array de vetores MD da condição intacta (N_samples, N_sensors).
        damaged_mds_dict (dict): Dicionário com vetores MD de cenários de dano.
                                 Ex: {'d_1': array, 'd_2': array}.
        sensor_names (list): Lista de nomes de sensores para os eixos do plot.
        floor_name (str): Nome do andar/grupo para o título do plot.
    """
    num_sensors = intact_mds_array.shape[1]
    if num_sensors < 2:
        print("❌ Visualização requer pelo menos 2 sensores no grupo.")
        return

    # Gera todas as combinações de pares de sensores
    sensor_pairs = list(combinations(range(num_sensors), 2))
    
    # Define o layout da grade de plots
    n_cols = 3
    n_rows = (len(sensor_pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    # Calcula a média dos MDs intactos para usar nas dimensões não plotadas
    mean_mds_intact = np.mean(intact_mds_array, axis=0)
    
    print(f"\n--- Gerando visualizações para {floor_name} ---")

    for i, (dim1, dim2) in enumerate(sensor_pairs):
        ax = axes[i]

        # Define a grade 2D para a avaliação do KDE
        x_min, x_max = intact_mds_array[:, dim1].min(), intact_mds_array[:, dim1].max()
        y_min, y_max = intact_mds_array[:, dim2].min(), intact_mds_array[:, dim2].max()
        x_pad = (x_max - x_min) * 0.2
        y_pad = (y_max - y_min) * 0.2
        xx, yy = np.mgrid[x_min - x_pad : x_max + x_pad : 100j, 
                          y_min - y_pad : y_max + y_pad : 100j]

        # Cria os pontos de avaliação N-dimensionais
        # As dimensões que não estão sendo plotadas são fixadas na sua média
        eval_points = np.tile(mean_mds_intact, (xx.ravel().shape[0], 1))
        eval_points[:, dim1] = xx.ravel()
        eval_points[:, dim2] = yy.ravel()

        # Avalia o KDE na grade e calcula a densidade
        log_density = kde_model.score_samples(eval_points)
        Z = np.exp(log_density).reshape(xx.shape)

        # Plota o contorno da densidade (heatmap)
        contour = ax.contourf(xx, yy, Z, levels=15, cmap='viridis_r')
        if i >= 2:  # Mostra colorbar apenas a partir do 3º gráfico
            fig.colorbar(contour, ax=ax, label='Densidade de Probabilidade')

        # Sobrepõe os pontos de dados
        # 1. Dados intactos (base para o KDE)
        ax.scatter(intact_mds_array[:, dim1], intact_mds_array[:, dim2], 
                   s=10, c='white', edgecolor='black', alpha=0.7, label='Intacto (Treino)')
                   
        # 2. Dados de dano
        colors = {'d_1': 'red', 'd_2': 'orange'}
        for scenario, mds_array in damaged_mds_dict.items():
            if mds_array.size > 0:
                ax.scatter(mds_array[:, dim1], mds_array[:, dim2], 
                           s=20, c=colors.get(scenario, 'magenta'), marker='X', label=f'Dano ({scenario})')

        ax.set_xlabel(f'MD {sensor_names[dim1]}')
        ax.set_ylabel(f'MD {sensor_names[dim2]}')
        ax.set_title(f'{sensor_names[dim1]} vs {sensor_names[dim2]}')
        if i == 0:  # Mostra legenda apenas no primeiro gráfico
            ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    # Oculta eixos não utilizados
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    #fig.suptitle(f'Visualização da Densidade do KDE para o Grupo "{floor_name}"', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Salva a figura
    plots_save_dir = os.path.join(os.path.dirname(ROOT_DATA_DIR), 'plots_kde_visualization')
    os.makedirs(plots_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"kde_density_{floor_name}_{timestamp}.png"
    plt.savefig(os.path.join(plots_save_dir, plot_filename), dpi=200, bbox_inches='tight')
    print(f"\n✅ Plot de visualização do KDE salvo em: {os.path.join(plots_save_dir, plot_filename)}")
    plt.show()


# ==============================================================================
# 3B. FUNÇÃO PARA GERAR 3 SUBPLOTS KDE (g2, g5, g6)
# ==============================================================================

def plot_kde_visualizations_multifloor(kde_models_dict, intact_mds_arrays, damaged_mds_dict_all, sensor_groups_by_floor, floors_to_plot=['g2', 'g5', 'g6']):
    """
    Gera uma figura com 3 subplots (um para cada andar/grupo) de contorno 2D KDE.

    Args:
        kde_models_dict (dict): Dicionário com modelos KDE treinados para cada andar.
                                Ex: {'g2': kde_model, 'g5': kde_model, 'g6': kde_model}
        intact_mds_arrays (dict): Dicionário com arrays de vetores MD intactos para cada andar.
        damaged_mds_dict_all (dict): Dicionário aninhado com vetores MD de dano para cada andar e cenário.
                                      Ex: {'g2': {'d_1': array, ...}, 'g5': {...}, ...}
        sensor_groups_by_floor (dict): Dicionário com grupos de sensores para cada andar.
        floors_to_plot (list): Lista de andares a plotar (ex: ['g2', 'g5', 'g6'])
    """
    
    num_floors = len(floors_to_plot)
    fig, axes = plt.subplots(1, num_floors, figsize=(num_floors * 8, 6), squeeze=False)
    axes = axes[0]  # Pega a primeira (e única) linha
    
    print(f"\n--- Gerando visualizações KDE para {floors_to_plot} ---")
    
    for idx, floor_name in enumerate(floors_to_plot):
        ax = axes[idx]
        
        if floor_name not in kde_models_dict:
            print(f"⚠️ Aviso: {floor_name} não encontrado nos dados processados.")
            continue
        
        kde_model = kde_models_dict[floor_name]
        intact_mds_array = intact_mds_arrays[floor_name]
        damaged_mds_dict = damaged_mds_dict_all.get(floor_name, {})
        sensors_in_floor = sensor_groups_by_floor[floor_name]
        
        num_sensors = intact_mds_array.shape[1]
        if num_sensors < 2:
            print(f"❌ {floor_name}: Visualização requer pelo menos 2 sensores.")
            continue
        
        # Usa o primeiro par de sensores para visualização
        dim1, dim2 = 0, 1
        
        # Define a grade 2D para a avaliação do KDE
        x_min, x_max = intact_mds_array[:, dim1].min(), intact_mds_array[:, dim1].max()
        y_min, y_max = intact_mds_array[:, dim2].min(), intact_mds_array[:, dim2].max()
        x_pad = (x_max - x_min) * 0.2
        y_pad = (y_max - y_min) * 0.2
        xx, yy = np.mgrid[x_min - x_pad : x_max + x_pad : 100j, 
                          y_min - y_pad : y_max + y_pad : 100j]
        
        # Cria os pontos de avaliação N-dimensionais
        mean_mds_intact = np.mean(intact_mds_array, axis=0)
        eval_points = np.tile(mean_mds_intact, (xx.ravel().shape[0], 1))
        eval_points[:, dim1] = xx.ravel()
        eval_points[:, dim2] = yy.ravel()
        
        # Avalia o KDE na grade e calcula a densidade
        log_density = kde_model.score_samples(eval_points)
        Z = np.exp(log_density).reshape(xx.shape)
        
        # Plota o contorno da densidade (heatmap)
        contour = ax.contourf(xx, yy, Z, levels=15, cmap='viridis_r')
        
        # Adiciona colorbar apenas no terceiro subplot (idx == 2, que é g6)
        if idx == 2:
            cbar = fig.colorbar(contour, ax=ax, label='Probability Density Function $p_g(.)$')
        
        # Sobrepõe os pontos de dados
        intact_label = 'Intact (train)' if idx == 2 else ''
        ax.scatter(intact_mds_array[:, dim1], intact_mds_array[:, dim2], 
                   s=10, c='white', edgecolor='black', alpha=0.7, label=intact_label)
        
        # Dados de dano - cores baseadas em scenario_colors, evitando amarelo e laranja
        colors = {
            'd_0_unknown': '#00BCD4',  # Ciano
            'd_1': '#EA4335',           # Vermelho
            'd_2': '#C2185B',           # Rosa/Magenta escuro (bem diferente do vermelho)
            'd_3': '#9C27B0',           # Roxo
            'd_4': '#1565C0',           # Azul escuro (substitui laranja)
            'd_5': '#34A853'            # Verde
        }
        legend_added = set()
        scenario_labels = {
            'd_0_unknown': 'Intact (test)',
            'd_1': 'Damage 1',
            'd_2': 'Damage 2',
            'd_3': 'Damage 3',
            'd_4': 'Damage 4',
            'd_5': 'Damage 5'
        }
        for scenario, mds_array in damaged_mds_dict.items():
            if mds_array.size > 0 and scenario != 'd_0_intact':
                label_text = scenario_labels.get(scenario, scenario) if idx == 2 else ''
                ax.scatter(mds_array[:, dim1], mds_array[:, dim2], 
                           s=20, c=colors.get(scenario, '#9C27B0'), marker='X', label=label_text, alpha=0.8)
                legend_added.add(scenario)
        
        ax.set_xlabel(f"MD {sensors_in_floor[dim1].replace('Sensor', 'Sensor ')}", fontsize=16)
        ax.set_ylabel(f"MD {sensors_in_floor[dim2].replace('Sensor', 'Sensor ')}", fontsize=16)
        subplot_titles = ['(a)', '(b)', '(c)']
        ax.set_title(subplot_titles[idx], fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Salva a figura
    plots_save_dir = os.path.join(os.path.dirname(ROOT_DATA_DIR), 'plots_kde_visualization')
    os.makedirs(plots_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"kde_density_multifloor_{timestamp}.png"
    plot_path = os.path.join(plots_save_dir, plot_filename)
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Plot de visualização KDE com 3 subplots salvo em: {plot_path}")
    plt.show()


# ==============================================================================
# 4. SCRIPT PRINCIPAL DE EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    
    TAMANHO_FONTE_BASE = 17
    plt.rcParams.update({
        # --- Configurações de Fonte ---
        'font.family': 'times new roman',                         # 1. Define a família da fonte para o texto NORMAL
        'mathtext.fontset': 'stix',                     # 2. <--- ADICIONADO: Garante que a fonte da FÓRMULA combine com a do texto
        
        # --- Configurações de Tamanho ---
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

    # --- ETAPA 1: Carregar o melhor modelo treinado ---
    db_path = os.path.join(OUTPUT_DIR, DB_FILENAME)
    storage_url = f"sqlite:///{db_path}"
    best_model_path, best_hyperparams = find_best_model_from_study(STUDY_NAME, storage_url, OUTPUT_DIR)
    
    if not best_model_path:
        exit()

    model = load_best_model(best_model_path, device, NUM_SENSORS_IN_TRAINED_MODEL, FIXED_PARAMS, best_hyperparams)
    if model is None:
        exit()

    # --- ETAPA 2: Processar os dados (igual ao script original) ---
    sensor_groups_by_floor = {
        'g1': ['Sensor1','Sensor2'],
        'g2': ['Sensor2','Sensor3'],
        'g3': ['Sensor3','Sensor4'],
        'g4': ['Sensor4','Sensor5'],
        'g5': ['Sensor5','Sensor6'],
        'g6': ['Sensor6','Sensor7']
    }
    
    # Valida que os grupos a visualizar estão configurados
    for floor in FLOORS_TO_VISUALIZE:
        if floor not in sensor_groups_by_floor:
            print(f"❌ Erro: '{floor}' não é um grupo de sensores válido.")
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

    # --- ETAPA 3: Preparar dados e treinar KDE para TODOS os andares (g2, g5, g6) ---
    kde_models_dict = {}
    intact_mds_arrays_dict = {}
    damaged_mds_dict_all = {}
    
    for floor_name in FLOORS_TO_VISUALIZE:
        print(f"\n--- Processando andar {floor_name} ---")
        sensors_in_floor = sensor_groups_by_floor[floor_name]
        
        # Coleta de dados intactos
        all_filenames_intact = set().union(*(sensor_mahalanobis_indicators[s].get(INTACT_CONDITION_NAME, {}).keys() for s in sensors_in_floor))
        floor_intact_mds_vectors = []
        for fname in sorted(list(all_filenames_intact)):
            md_vector = [sensor_mahalanobis_indicators[s][INTACT_CONDITION_NAME].get(fname) for s in sensors_in_floor]
            if all(v is not None for v in md_vector):
                floor_intact_mds_vectors.append(md_vector)
        intact_mds_array_floor = np.array(floor_intact_mds_vectors)
        
        if intact_mds_array_floor.shape[0] < 5:
            print(f"❌ Dados intactos insuficientes para {floor_name}. Pulando.")
            continue
        
        intact_mds_arrays_dict[floor_name] = intact_mds_array_floor
        
        # Coleta de dados de dano para TODOS os cenários (não apenas d_0_unknown, d_1, d_2)
        damaged_mds_floor = {}
        for scenario in DAMAGE_SCENARIOS:
            if scenario == INTACT_CONDITION_NAME:
                continue
            all_filenames_damage = set().union(*(sensor_mahalanobis_indicators[s].get(scenario, {}).keys() for s in sensors_in_floor))
            floor_damage_mds_vectors = []
            for fname in sorted(list(all_filenames_damage)):
                md_vector = [sensor_mahalanobis_indicators[s][scenario].get(fname) for s in sensors_in_floor]
                if all(v is not None for v in md_vector):
                    floor_damage_mds_vectors.append(md_vector)
            if floor_damage_mds_vectors:
                damaged_mds_floor[scenario] = np.array(floor_damage_mds_vectors)
            else:
                damaged_mds_floor[scenario] = np.array([])
        
        damaged_mds_dict_all[floor_name] = damaged_mds_floor
        
        # Treinamento do KDE para este andar
        kde_model_floor = KernelDensity(kernel='gaussian', bandwidth='scott')
        kde_model_floor.fit(intact_mds_array_floor)
        kde_models_dict[floor_name] = kde_model_floor
        print(f"  ✓ KDE treinado. Bandwidth: {kde_model_floor.bandwidth_:.4f}")
    
    # --- ETAPA 4: Gerar visualização com 3 subplots (g2, g5, g6) ---
    if len(kde_models_dict) == len(FLOORS_TO_VISUALIZE):
        plot_kde_visualizations_multifloor(
            kde_models_dict=kde_models_dict,
            intact_mds_arrays=intact_mds_arrays_dict,
            damaged_mds_dict_all=damaged_mds_dict_all,
            sensor_groups_by_floor=sensor_groups_by_floor,
            floors_to_plot=FLOORS_TO_VISUALIZE
        )
    else:
        print(f"❌ Nem todos os andares foram processados com sucesso.")