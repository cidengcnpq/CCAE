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
import pandas as pd
from datetime import datetime
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import optuna

# ==============================================================================
# CONFIGURATION SECTION (TRADITIONAL CAE)
# ==============================================================================

# --- Optuna Study Directory Settings ---
BASE_DIR_OUTPUT = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo"
# Traditional CAE file
OUTPUT_DIR = os.path.join(BASE_DIR_OUTPUT, 'optuna_optimization_output_Z24_CAE')
STUDY_NAME = "cae_hyperparam_optimization_Z24"
DB_FILENAME = "cae_optimization_Z24.db"

# --- Data Analysis Directory Settings ---
ROOT_DATA_DIR = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo/imagens_CWT_Z24_v1_matlab"

# --- Fixed Model Settings  ---
FIXED_PARAMS = {
    'bottleneck_channels': 4,
    # loss/classifier parameters removed, do not exist in CAE
}

# --- Dataset and Analysis Settings ---
ALL_SENSORS_MODEL_TRAINED_ON = [f'Sensor{i}' for i in range(1, 8)]
NUM_SENSORS_IN_TRAINED_MODEL = len(ALL_SENSORS_MODEL_TRAINED_ON)
DAMAGE_SCENARIOS = ['d_0_intact', 'd_0_unknown', 'd_1', 'd_2','d_3','d_4', 'd_5']
INTACT_CONDITION_NAME = 'd_0_intact'


# ==============================================================================
# 1. FUNCTION TO LOAD THE BEST OPTUNA STUDY MODEL
# ==============================================================================
def find_best_model_from_study(study_name, storage_url, output_dir):
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        best_trial = study.best_trial

        print("--- Melhor Trial Encontrado (CAE) ---")
        print(f"  Número do Trial: {best_trial.number}")
        print(f"  Valor (Loss Média): {best_trial.value:.5f}")
        print("  Melhores Hiperparâmetros:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        chosen_fold = 1
        model_path = os.path.join(output_dir, f'trial_{best_trial.number}', f'best_model_fold_{chosen_fold}.pth')

        if not os.path.exists(model_path):
            print(f"❌ Erro: Arquivo do modelo não encontrado em: {model_path}")
            return None, None

        return model_path, best_trial.params

    except Exception as e:
        print(f"❌ Erro ao carregar o estudo Optuna: {e}")
        return None, None

# ==============================================================================
# 2. DEFINITION OF THE TRADITIONAL CAE MODEL
# ==============================================================================
class TraditionalCAE(nn.Module):
    def __init__(self, bottleneck_channels):
        super(TraditionalCAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_channels), nn.ReLU()
        )
        
        # Decoder (without embedding)
        decoder_input_channels = bottleneck_channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(decoder_input_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        encoded_features = self.bottleneck_conv(x)
        x_reconstructed = self.decoder(encoded_features)
        return x_reconstructed

    def get_latent_features(self, x):
        """Extrai as features do bottleneck e achata para vetor 1D."""
        with torch.no_grad():
            x_encoded = self.encoder(x)
            encoded_features = self.bottleneck_conv(x_encoded)
            # Applies Global Average Pooling to flatten a 1D vector.
            pooled_latent = nn.AdaptiveAvgPool2d((1, 1))(encoded_features)
            return pooled_latent.view(pooled_latent.size(0), -1)

# ==============================================================================
# 3. Definitions of Dataset, Data Loader, and Auxiliary Functions
# ==============================================================================
class CWTDatasetAnalysis(Dataset):
    def __init__(self, root_folder, all_sensors_for_mapping):
        self.image_paths = glob.glob(os.path.join(root_folder, '**', '*.png'), recursive=True)
        
        if not self.image_paths:
            print(f"Aviso: Nenhuma imagem PNG encontrada em: {root_folder}")

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
            print(f"Erro ao carregar imagem {img_path}: {e}. Retornando None.")
            return None, None

def collate_fn_analysis(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return torch.tensor([]), []
    images, names = zip(*batch)
    return torch.stack(images), list(names)

def load_all_sensor_data(root_dir, damage_scenarios, sensors_for_analysis):
    all_data = {sensor: {} for sensor in sensors_for_analysis}
    for sensor_name in sensors_for_analysis:
        for condition in damage_scenarios:
            path = os.path.join(root_dir, condition, sensor_name)
            if os.path.isdir(path):
                dataset = CWTDatasetAnalysis(path, [])
                if len(dataset) > 0:
                    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_analysis)
                    all_data[sensor_name][condition] = dataloader
            else:
                print(f"Aviso: Pasta não encontrada: {path}. Pulando.")
    return all_data

def extract_latent_features(model, dataloader, device):
    """
    Extrai features latentes para CAE (sem scaling factor).
    """
    model.eval()
    features_dict = {}
    with torch.no_grad():
        for images, names in dataloader:
            if images.numel() == 0: continue
            images = images.to(device)
            
            # Chamada direta sem scaling
            latent_features = model.get_latent_features(images).cpu().numpy()
            
            for i, name in enumerate(names):
                features_dict[name] = latent_features[i]
    return features_dict

def calculate_mahalanobis_distance(features, mean, inv_cov):
    if not features: return np.array([])
    features_array = np.array(features)
    if features_array.ndim == 1: features_array = features_array.reshape(1, -1)
    distances = [mahalanobis(f, mean, inv_cov) for f in features_array]
    return np.array(distances)

def load_best_model(model_path, device, fixed_params):
    """Instancia o TraditionalCAE e carrega pesos."""
    model = TraditionalCAE(
        bottleneck_channels=fixed_params['bottleneck_channels']
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Modelo carregado com sucesso de: {model_path}")
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo de {model_path}: {e}")
        return None

# ==============================================================================
# 4. MAIN FUNCTION FOR ANALYSIS AND PLOTTING
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # --- STEP 1: Find and upload the best model ---
    db_path = os.path.join(OUTPUT_DIR, DB_FILENAME)
    storage_url = f"sqlite:///{db_path}"
    
    best_model_path, best_hyperparams = find_best_model_from_study(STUDY_NAME, storage_url, OUTPUT_DIR)
    
    if not best_model_path:
        return

    model = load_best_model(
        model_path=best_model_path,
        device=device,
        fixed_params=FIXED_PARAMS
    )
    if model is None: return

    # --- STEP 2: Run the anomaly analysis ---
    sensor_groups_by_floor = {
        'g1': ['Sensor1','Sensor2'],
        'g2': ['Sensor2','Sensor3'],
        'g3': ['Sensor3','Sensor4'],
        'g4': ['Sensor4','Sensor5'],
        'g5': ['Sensor5','Sensor6'],
        'g6': ['Sensor6','Sensor7']
    }
    all_sensors_for_analysis = [s for floor_sensors in sensor_groups_by_floor.values() for s in floor_sensors]

    print("\n--- Carregando dados para análise ---")
    all_sensor_data = load_all_sensor_data(ROOT_DATA_DIR, DAMAGE_SCENARIOS, all_sensors_for_analysis)

    # NOTE: The scaling_factor calculation has been removed as it does not apply to traditional CAE.
        
    print("\n--- Extraindo features da camada latente (CAE) ---")
    all_sensor_latent_features = {s: {c: {} for c in DAMAGE_SCENARIOS} for s in all_sensors_for_analysis}
    for sensor_name, conditions in all_sensor_data.items():
        print(f"Processando sensor: {sensor_name}")
        for condition, dataloader in conditions.items():
            if dataloader:
                # Adapted function (without scaling factor)
                features_dict = extract_latent_features(model, dataloader, device)
                all_sensor_latent_features[sensor_name][condition].update(features_dict)

    print("\n--- Calculando Distância de Mahalanobis por sensor ---")
    sensor_mahalanobis_indicators = {s: {c: {} for c in DAMAGE_SCENARIOS} for s in all_sensors_for_analysis}

    for sensor_name in all_sensors_for_analysis:
        intact_features = list(all_sensor_latent_features[sensor_name].get(INTACT_CONDITION_NAME, {}).values())
        if not intact_features:
            print(f"Aviso: Nenhuma feature intacta para {sensor_name}. Pulando Mahalanobis.")
            continue
        
        intact_features_array = np.array(intact_features)
        if intact_features_array.shape[0] <= 1:
            print(f"Aviso: Dados insuficientes para calcular covariância para {sensor_name}. Pulando.")
            continue
        
        try:
            variances = np.var(intact_features_array, axis=0)
            non_zero_variance_indices = np.where(variances > 1e-9)[0]
            if non_zero_variance_indices.size == 0:
                print(f"Aviso: Todas as features têm variância zero para {sensor_name}. Pulando.")
                continue

            intact_features_filtered = intact_features_array[:, non_zero_variance_indices]
            mean_feature = np.mean(intact_features_filtered, axis=0)
            cov = EmpiricalCovariance().fit(intact_features_filtered).covariance_
            inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

            for condition in DAMAGE_SCENARIOS:
                features_condition = list(all_sensor_latent_features[sensor_name].get(condition, {}).values())
                filenames_condition = list(all_sensor_latent_features[sensor_name].get(condition, {}).keys())
                if features_condition:
                    features_cond_filtered = [f[non_zero_variance_indices] for f in features_condition]
                    distances = calculate_mahalanobis_distance(features_cond_filtered, mean_feature, inv_cov)
                    for filename, dist in zip(filenames_condition, distances):
                        sensor_mahalanobis_indicators[sensor_name][condition][filename] = dist
        except np.linalg.LinAlgError:
            print(f"Erro de Álgebra Linear (matriz singular) para o sensor {sensor_name}. Pulando.")
        except Exception as e:
            print(f"Erro inesperado no cálculo de Mahalanobis para {sensor_name}: {e}")

    print("\n--- Fusão multisensor usando KDE POR ANDAR (GROUPS) ---")
    floor_damage_indicators = {floor: {c: {} for c in DAMAGE_SCENARIOS} for floor in sensor_groups_by_floor.keys()}
    
    for floor, sensors_in_floor in sensor_groups_by_floor.items():
        print(f"Processando o grupo: {floor}")
        
        all_filenames_intact = set()
        for s in sensors_in_floor:
            all_filenames_intact.update(sensor_mahalanobis_indicators[s].get(INTACT_CONDITION_NAME, {}).keys())
        all_filenames_intact = sorted(list(all_filenames_intact))
        floor_intact_mds_vectors = []
        for filename in all_filenames_intact:
            mds_vector = [sensor_mahalanobis_indicators[s].get(INTACT_CONDITION_NAME, {}).get(filename) for s in sensors_in_floor]
            if all(v is not None for v in mds_vector):
                floor_intact_mds_vectors.append(mds_vector)
        
        intact_mds_array_floor = np.array(floor_intact_mds_vectors)

        if intact_mds_array_floor.shape[0] < 5:
            print(f"  Aviso: Dados intactos insuficientes ({intact_mds_array_floor.shape[0]} amostras) para treinar KDE para o {floor}.")
            continue
            
        try:
            grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                {'bandwidth': 10**np.linspace(-2, 2, 100)},
                                cv=min(5, intact_mds_array_floor.shape[0]))
            grid.fit(intact_mds_array_floor)
            kde_model_floor = grid.best_estimator_
            print(f"  Melhor Bandwidth (KDE) para {floor}: {kde_model_floor.bandwidth:.4f}")

            for condition in DAMAGE_SCENARIOS:
                all_filenames_condition = set()
                for s in sensors_in_floor:
                    all_filenames_condition.update(sensor_mahalanobis_indicators[s].get(condition, {}).keys())
                
                for filename in all_filenames_condition:
                    mds_vector = [sensor_mahalanobis_indicators[s].get(condition, {}).get(filename) for s in sensors_in_floor]
                    if all(v is not None for v in mds_vector):
                        mds_np_vector = np.array(mds_vector).reshape(1, -1)
                        global_di_kde = -kde_model_floor.score_samples(mds_np_vector)[0]
                        floor_damage_indicators[floor][condition][filename] = global_di_kde
        except Exception as e:
            print(f"  Erro ao treinar ou aplicar KDE para o {floor}: {e}")

    print("\n--- Gerando gráfico de resultados ---")
    df_rows_boxplot = []
    for floor, conditions_data in floor_damage_indicators.items():
        for condition, filenames_data in conditions_data.items():
            for filename, di in filenames_data.items():
                if di > 0:
                    df_rows_boxplot.append({'andar': floor, 'cenario': condition, 'di': di})

    if not df_rows_boxplot:
        print("❌ Nenhum dado de Índice de Dano para plotar. A análise pode ter falhado.")
        return
        
    df_boxplot = pd.DataFrame(df_rows_boxplot)
    floor_order = list(sensor_groups_by_floor.keys())
    df_boxplot['andar'] = pd.Categorical(df_boxplot['andar'], categories=floor_order, ordered=True)
    df_boxplot['cenario'] = pd.Categorical(df_boxplot['cenario'], categories=DAMAGE_SCENARIOS, ordered=True)
    df_boxplot = df_boxplot.sort_values(by=['andar', 'cenario'])

    floor_thresholds = {}
    for floor in floor_order:
        intact_di_values = df_boxplot[(df_boxplot['andar'] == floor) & (df_boxplot['cenario'] == INTACT_CONDITION_NAME)]['di']
        if not intact_di_values.empty:
            threshold = np.percentile(intact_di_values, 99)
            floor_thresholds[floor] = threshold
            print(f"Threshold de Anomalia para {floor}: {threshold:.4f}")

    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Define the new color palette
    colors = {
        'd_0_intact': '#FBBC05',
        'd_0_unknown': '#00BCD4',
        'd_1': '#EA4335',
        'd_2': '#C2185B',
        'd_3': '#9C27B0',
        'd_4': '#1565C0',
        'd_5': '#34A853',
    }
    
    sns.boxplot(data=df_boxplot, x='andar', y='di', hue='cenario', ax=ax,
                palette=colors, showfliers=False)
    
    for i, floor in enumerate(floor_order):
        if floor in floor_thresholds:
            ax.hlines(y=floor_thresholds[floor], xmin=i - 0.4, xmax=i + 0.4, 
                      colors='purple', linestyles='--', label='Threshold (99%)' if i == 0 else "")

    ax.set_xlabel('Groups of sensors', fontsize=14, fontfamily='times new roman')
    ax.set_ylabel('Global Damage Index (DI-KDE) [log-scale]', fontsize=14, fontfamily='times new roman')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.6)
    
    # --- Axis Adjustment (Ticks) ---
    tick_labels = ['g\u2081', 'g\u2082', 'g\u2083', 'g\u2084', 'g\u2085', 'g\u2086']
    ax.set_xticklabels(tick_labels, fontfamily='times new roman', fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=10, fontfamily='times new roman', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    
    # Create the legend 
    scenario_labels = {
        'd_0_intact': 'Intact (train)',
        'd_0_unknown': 'Intact (test)',
        'd_1': 'Damage 1',
        'd_2': 'Damage 2',
        'd_3': 'Damage 3',
        'd_4': 'Damage 4',
        'd_5': 'Damage 5'
    }
    
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [scenario_labels.get(label, label) for label in labels]
    legend = ax.legend(handles, new_labels, title='Damage Scenario', bbox_to_anchor=(1.02, 1), loc='upper left',fontsize=14)
    
    legend.get_title().set_fontfamily('times new roman')
    legend.get_title().set_fontsize(14)
    for text in legend.get_texts():
        text.set_fontfamily('times new roman')
        text.set_fontsize(14)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save CAE file
    plots_save_dir = os.path.join(os.path.dirname(ROOT_DATA_DIR), 'plots_analise_melhor_modelo_CAE')
    os.makedirs(plots_save_dir, exist_ok=True)
    plot_filename = f"boxplot_DI_global_CAE_Z24_{timestamp}.png"
    plt.savefig(os.path.join(plots_save_dir, plot_filename), dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot salvo em: {os.path.join(plots_save_dir, plot_filename)}")
    plt.show()
    
    # --- STEP 3: Calculation of Detection Metrics ---

    def calculate_detection_stats(df_boxplot, floor_thresholds):
        results = []
        for (floor, scenario), group in df_boxplot.groupby(['andar', 'cenario']):
            if floor in floor_thresholds:
                threshold = floor_thresholds[floor]
                total_samples = len(group)
                below_threshold = (group['di'] <= threshold).sum()
                percentage = (below_threshold / total_samples) * 100
                
                results.append({
                    'Group': floor,
                    'Scenario': scenario,
                    'Samples_Below_Thr': below_threshold,
                    'Total_Samples': total_samples,
                    'Percentage_Below (%)': round(percentage, 2)
                })

        stats_df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("PORCENTAGEM DE AMOSTRAS ABAIXO DO THRESHOLD (INTACT) - TRADITIONAL CAE")
        print("="*60)
        print(stats_df.to_string(index=False))
        print("-" * 60)
        
        return stats_df

    calculate_detection_stats(df_boxplot, floor_thresholds)


    # --- STEP 4: Calculating Metrics for the Article Text ---

    def print_paper_metrics(df):
        # Define which groups belong to each side (adjust according to the visual result of the CAE)
        utzenstorf_groups = ['g1', 'g2', 'g3']      
        koppigen_groups = ['g4', 'g5', 'g6']    
        
        damage_scenarios = ['d_1', 'd_2', 'd_3', 'd_4', 'd_5']
        intact_test_scenario = 'd_0_unknown'

        # Filters only data from damage scenarios
        df_damage = df[df['cenario'].isin(damage_scenarios)]
        
        # 1. Calculation of X: Spatial Comparison (Damaged Side vs. Healthy Side)
        mean_di_koppigen = df_damage[df_damage['andar'].isin(koppigen_groups)]['di'].mean()
        mean_di_utzenstorf = df_damage[df_damage['andar'].isin(utzenstorf_groups)]['di'].mean()
        
        X = mean_di_koppigen / mean_di_utzenstorf if mean_di_utzenstorf > 0 else 0

        # 2. Calculation of Y: Condition Comparison (Damage vs. Intact Test on the affected side)
        df_intact = df[df['cenario'] == intact_test_scenario]
        mean_di_koppigen_intact = df_intact[df_intact['andar'].isin(koppigen_groups)]['di'].mean()
        
        Y = mean_di_koppigen / mean_di_koppigen_intact if mean_di_koppigen_intact > 0 else 0

        print("-" * 60)
        print("VALORES PARA O TEXTO DO ARTIGO (CAE Results)")
        print("-" * 60)
        print(f"Grupos definidos como Koppigen (Dano): {koppigen_groups}")
        print(f"Grupos definidos como Utzenstorf:      {utzenstorf_groups}")
        print("-" * 60)
        print(f"Média DI (Koppigen side, Damage Scenarios): {mean_di_koppigen:.4f}")
        print(f"Média DI (Utzenstorf side, Damage Scenarios): {mean_di_utzenstorf:.4f}")
        print(f"Média DI (Koppigen side, Intact Test): {mean_di_koppigen_intact:.4f}")
        print("-" * 60)
        print(f"X (Razão Espacial): {X:.2f}")
        print(f"Y (Razão Dano/Intacto): {Y:.2f}")
        print("-" * 60)
        print("\nFRASE PREENCHIDA:")
        print(f"...are, on average, {X:.1f} times higher than those on the Utzenstorf side...")

    print_paper_metrics(df_boxplot)

if __name__ == "__main__":

    main()
