import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
import pandas as pd
from datetime import datetime
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import optuna

# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================

# --- Optuna Study Directory Settings ---
# Adjust the BASE_DIR to the parent directory of 'optuna_optimization_output'.
BASE_DIR_OUTPUT = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo"
OUTPUT_DIR = os.path.join(BASE_DIR_OUTPUT, 'optuna_optimization_output_yellow_frame')
STUDY_NAME = "ccae_hyperparam_optimization_yellow_frame"
DB_FILENAME = "ccae_optimization_yellow_frame.db"

# --- Data Analysis Directory Settings ---
ROOT_DATA_DIR = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo/imagens_cwt_yellow_frame"

# --- Fixed Model Settings (IDENTICAL TO TRAINING) ---
FIXED_PARAMS = {
    'classifier_linear_dim': 64,
    'bottleneck_channels': 4,
    'reconstruction_loss_weight': 1.0,
    'classification_loss_weight': 0.1,
}

# --- Dataset and Analysis Settings ---
ALL_SENSORS_MODEL_TRAINED_ON = [f'Sensor{i}' for i in range(1, 16)]
NUM_SENSORS_IN_TRAINED_MODEL = len(ALL_SENSORS_MODEL_TRAINED_ON)
DAMAGE_SCENARIOS = ['d_0_intact', 'd_0_unknown', 'd_1', 'd_2']
INTACT_CONDITION_NAME = 'd_0_intact'


# ==============================================================================
# 1. FUNCTION TO LOAD THE BEST OPTUNA STUDY MODEL
# ==============================================================================
def find_best_model_from_study(study_name, storage_url, output_dir):
    """Loads an Optuna study, finds the best trial, and returns the path to the first-fold model and its hyperparameters."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        best_trial = study.best_trial

        print("--- Melhor Trial Encontrado ---")
        print(f"  Número do Trial: {best_trial.number}")
        print(f"  Valor (Loss Média): {best_trial.value:.5f}")
        print("  Melhores Hiperparâmetros:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        chosen_fold = 1
        model_path = os.path.join(output_dir, f'trial_{best_trial.number}', f'best_model_fold_{chosen_fold}.pth')

        if not os.path.exists(model_path):
            print(f"❌ Erro: Arquivo do modelo não encontrado em: {model_path}")
            print("Verifique se o OUTPUT_DIR está correto e se o trial foi concluído.")
            return None, None

        return model_path, best_trial.params

    except Exception as e:
        print(f"❌ Erro ao carregar o estudo Optuna: {e}")
        print(f"Verifique se o nome do estudo ('{study_name}') e o caminho do banco de dados ('{storage_url}') estão corretos.")
        return None, None

# ==============================================================================
# 2. DEFINITION OF THE CCAE MODEL (IDENTICAL TO THAT USED IN TRAINING)
# ==============================================================================
class CCAE(nn.Module):
    def __init__(self, num_sensors, bottleneck_channels, classifier_linear_dim, dropout_rate):
        super(CCAE, self).__init__()
        self.num_sensors = num_sensors

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
        
        # Embedding
        self.sensor_embedding = nn.Embedding(num_sensors, 64) 

        # Classifier
        classifier_input_dim = bottleneck_channels * 8 * 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_input_dim, classifier_linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_linear_dim, num_sensors)
        )

        # Decoder
        decoder_input_channels = bottleneck_channels + 1
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
        sensor_logits = self.classifier(encoded_features)
        sensor_probabilities = torch.softmax(sensor_logits, dim=1)
        all_sensor_embeddings = self.sensor_embedding(torch.arange(self.num_sensors).to(x.device))
        combined_sensor_emb = (sensor_probabilities.unsqueeze(2) * all_sensor_embeddings.unsqueeze(0)).sum(dim=1)
        combined_sensor_emb_reshaped = combined_sensor_emb.view(-1, 1, 8, 8)
        concatenated_features = torch.cat([encoded_features, combined_sensor_emb_reshaped], dim=1)
        x_reconstructed = self.decoder(concatenated_features)
        return x_reconstructed, sensor_logits

    def get_latent_features(self, x, scaling_factor):
        """Extracts, scales, and flattens conditional latent features into a 1D vector."""
        with torch.no_grad():
            x_encoded = self.encoder(x)
            encoded_features = self.bottleneck_conv(x_encoded)
            sensor_logits = self.classifier(encoded_features)
            sensor_probabilities = torch.softmax(sensor_logits, dim=1)
            all_sensor_embeddings = self.sensor_embedding(torch.arange(self.num_sensors).to(x.device))
            combined_sensor_emb = (sensor_probabilities.unsqueeze(2) * all_sensor_embeddings.unsqueeze(0)).sum(dim=1)
            
            # APPLICATION OF THE SCALE FACTOR
            combined_sensor_emb_reshaped = combined_sensor_emb.view(-1, 1, 8, 8) * scaling_factor
            
            concatenated_features = torch.cat([encoded_features, combined_sensor_emb_reshaped], dim=1)
            
            # Applies Global Average Pooling to flatten a 1D vector.
            pooled_latent = nn.AdaptiveAvgPool2d((1, 1))(concatenated_features)
            
            return pooled_latent.view(pooled_latent.size(0), -1)

# ==============================================================================
# 3. Definitions of Dataset, Data Loader, and Auxiliary Functions
# ==============================================================================
class CWTDatasetAnalysis(Dataset):
    def __init__(self, root_folder, all_sensors_for_mapping):
        self.image_paths = glob.glob(os.path.join(root_folder, '**', '*.png'), recursive=True)
        self.sensor_to_idx = {sensor: i for i, sensor in enumerate(all_sensors_for_mapping)}
        
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

def load_all_sensor_data(root_dir, damage_scenarios, sensors_for_analysis, all_sensors_for_mapping):
    all_data = {sensor: {} for sensor in sensors_for_analysis}
    for sensor_name in sensors_for_analysis:
        for condition in damage_scenarios:
            path = os.path.join(root_dir, condition, sensor_name)
            if os.path.isdir(path):
                dataset = CWTDatasetAnalysis(path, all_sensors_for_mapping)
                if len(dataset) > 0:
                    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_analysis)
                    all_data[sensor_name][condition] = dataloader
            else:
                print(f"Aviso: Pasta não encontrada: {path}. Pulando.")
    return all_data

def extract_latent_features(model, dataloader, device, scaling_factor):
    model.eval()
    features_dict = {}
    with torch.no_grad():
        for images, names in dataloader:
            if images.numel() == 0: continue
            images = images.to(device)
             # Pass the scaling factor to the model method.
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
    """Instancia o modelo com os parâmetros corretos e carrega os pesos."""
    model = CCAE(
        num_sensors=num_sensors,
        bottleneck_channels=fixed_params['bottleneck_channels'],
        classifier_linear_dim=fixed_params['classifier_linear_dim'],
        dropout_rate=best_hyperparams['dropout_rate']
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
# 4. FUNCTION FOR CALCULATING THE RESCALING FACTOR WITH MULTIPLE SENSORS
# ==============================================================================
def calculate_scaling_factor_all_sensors(model, dataloaders, device):
    """
   Estimate a scaling factor to balance 'encoded_features' and 'combined_sensor_emb_reshaped' using data from all intact dataloaders.
    """
    model.eval()
    encoded_norms = []
    sensor_emb_norms = []
    
    print("Estimando fator de escala a partir dos dados intactos de todos os sensores...")
    
    with torch.no_grad():
        for dataloader in dataloaders:
            for images, _ in dataloader:
                if images.numel() == 0:
                    continue
                images = images.to(device)
                
                # Forward pass to obtain the intermediate features.
                x_encoded = model.encoder(images)
                encoded_features = model.bottleneck_conv(x_encoded)
                sensor_logits = model.classifier(encoded_features)
                sensor_probabilities = torch.softmax(sensor_logits, dim=1)
                all_sensor_embeddings = model.sensor_embedding(
                    torch.arange(model.num_sensors).to(device)
                )
                combined_sensor_emb = (
                    sensor_probabilities.unsqueeze(2) * all_sensor_embeddings.unsqueeze(0)
                ).sum(dim=1)
                combined_sensor_emb_reshaped = combined_sensor_emb.view(-1, 1, 8, 8)

                # Calculate the Frobenius norm for each feature in the batch.
                encoded_norm = torch.linalg.norm(encoded_features.flatten(start_dim=1), dim=1)
                sensor_emb_norm = torch.linalg.norm(combined_sensor_emb_reshaped.flatten(start_dim=1), dim=1)
                
                encoded_norms.append(encoded_norm.cpu().numpy())
                sensor_emb_norms.append(sensor_emb_norm.cpu().numpy())
            
    if not encoded_norms or not sensor_emb_norms:
        print("Aviso: Nenhum dado encontrado para calcular o fator de escala. Retornando 1.0.")
        return 1.0
        
    avg_encoded_norm = np.mean(np.concatenate(encoded_norms))
    avg_sensor_emb_norm = np.mean(np.concatenate(sensor_emb_norms))
    
    if avg_sensor_emb_norm == 0:
        print("Aviso: Média da norma da feature do sensor é zero. Retornando 1.0.")
        return 1.0

    scaling_factor = avg_encoded_norm / avg_sensor_emb_norm
    print(f"Média da Norma 'encoded_features': {avg_encoded_norm:.4f}")
    print(f"Média da Norma 'combined_sensor_emb_reshaped': {avg_sensor_emb_norm:.4f}")
    print(f"Fator de Escala Estimado: {scaling_factor:.4f}")
    
    return scaling_factor

# ==============================================================================
# 5. MAIN FUNCTION FOR ANALYSIS AND PLOTTING
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
        num_sensors=NUM_SENSORS_IN_TRAINED_MODEL,
        fixed_params=FIXED_PARAMS,
        best_hyperparams=best_hyperparams
    )
    if model is None: return

    # --- STEP 2: Run the anomaly analysis ---
    sensor_groups_by_floor = {
        'Base': ['Sensor1', 'Sensor2', 'Sensor3'],
        '1st floor': ['Sensor4', 'Sensor5', 'Sensor6'],
        '2nd floor': ['Sensor7', 'Sensor8', 'Sensor9'],
        '3rd floor': ['Sensor10', 'Sensor11', 'Sensor12'],
        '4th floor': ['Sensor13', 'Sensor14', 'Sensor15']
    }
    all_sensors_for_analysis = [s for floor_sensors in sensor_groups_by_floor.values() for s in floor_sensors]

    print("\n--- Carregando dados para análise ---")
    all_sensor_data = load_all_sensor_data(ROOT_DATA_DIR, DAMAGE_SCENARIOS, all_sensors_for_analysis, ALL_SENSORS_MODEL_TRAINED_ON)

    # --- Calculating the scaling factor using all the data intact. ---
    print("\n--- Calculando fator de escala usando todos os dados intactos ---")
    
    all_intact_dataloaders = []
    for sensor_name in all_sensors_for_analysis:
        if INTACT_CONDITION_NAME in all_sensor_data[sensor_name]:
            dataloader = all_sensor_data[sensor_name][INTACT_CONDITION_NAME]
            if dataloader and len(dataloader.dataset) > 0:
                all_intact_dataloaders.append(dataloader)

    if all_intact_dataloaders:
        scaling_factor = calculate_scaling_factor_all_sensors(model, all_intact_dataloaders, device)
    else:
        print("❌ Aviso: Nenhum dado intacto encontrado em nenhum sensor. Usando fator de 1.0.")
        scaling_factor = 1.0
        
    print("\n--- Extraindo features da camada latente ---")
    all_sensor_latent_features = {s: {c: {} for c in DAMAGE_SCENARIOS} for s in all_sensors_for_analysis}
    for sensor_name, conditions in all_sensor_data.items():
        print(f"Processando sensor: {sensor_name}")
        for condition, dataloader in conditions.items():
            if dataloader:
                features_dict = extract_latent_features(model, dataloader, device, scaling_factor)
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

    print("\n--- Fusão multisensor usando KDE POR ANDAR ---")
    floor_damage_indicators = {floor: {c: {} for c in DAMAGE_SCENARIOS} for floor in sensor_groups_by_floor.keys()}
    
    for floor, sensors_in_floor in sensor_groups_by_floor.items():
        print(f"Processando o andar: {floor}")
        
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
            
    # Rename scenario legends
    labels_legenda = {
        'd_0_intact': 'Intact (train)',
        'd_0_unknown': 'Intact (test)',
        'd_1': 'Damage 1',
        'd_2': 'Damage 2'
    }

    # 2. Update the DataFrame with the new names
    df_boxplot['cenario'] = df_boxplot['cenario'].map(labels_legenda)

    # 3. Define the new color palette
    nova_palette = {
        'Intact (train)': '#00BCD4',  
        'Intact (test)': '#34A853', 
        'Damage 1': '#EA4335',          
        'Damage 2': '#FBBC05'           
    }
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.boxplot(data=df_boxplot, x='andar', y='di', hue='cenario', ax=ax,
                    palette=nova_palette, 
                    showfliers=False)
    
    # --- Axis Adjustment (Ticks) ---
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)
    
    for i, floor in enumerate(floor_order):
        if floor in floor_thresholds:
            ax.hlines(y=floor_thresholds[floor], xmin=i - 0.4, xmax=i + 0.4, 
                      colors='purple', linestyles='--', label='Threshold (99%)' if i == 0 else "")

    ax.set_xlabel('Group of sensors', fontsize=14, font='Times New Roman')
    ax.set_ylabel('Global damage index (DI-KDE) [log-scale]', fontsize=14, font='Times New Roman')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.6)
    plt.xticks(rotation=10)
    # 1. Create the legend 
    legend = plt.legend(title='Damage scenario', 
                        bbox_to_anchor=(1.02, 1), 
                        loc='upper left',
                        prop={'family': 'Times New Roman', 'size': 12})

    # 2. change in the font of the caption TITLE.
    plt.setp(legend.get_title(), fontname='Times New Roman', fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_save_dir = os.path.join(os.path.dirname(ROOT_DATA_DIR), 'plots_analise_melhor_modelo')
    os.makedirs(plots_save_dir, exist_ok=True)
    plot_filename = f"boxplot_DI_global_{timestamp}.png"
    plt.savefig(os.path.join(plots_save_dir, plot_filename), dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot salvo em: {os.path.join(plots_save_dir, plot_filename)}")
    plt.show()


    # --- STEP 3: Calculation of Detection Metrics ---

    def calculate_detection_stats(df_boxplot, floor_thresholds):
        """
        Calculates the percentage of samples below the threshold (classified as intact).
        """
        results = []
        
        # Group by group (floor) and setting.
        for (floor, scenario), group in df_boxplot.groupby(['andar', 'cenario']):
            if floor in floor_thresholds:
                threshold = floor_thresholds[floor]
                total_samples = len(group)
                # Samples below the threshold are considered "Negative" (non-anomalous).
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
        print("PORCENTAGEM DE AMOSTRAS ABAIXO DO THRESHOLD (INTACT)")
        print("="*60)
        print(stats_df.to_string(index=False))
        print("-" * 60)
        print("Nota: Para cenários de dano (d_1 a d_5), valores baixos indicam alta detecção.")
        print("Para cenários intactos, valores altos indicam baixa taxa de falso alarme.")
        
        return stats_df

    # Para usar, basta chamar a função logo após a definição de 'floor_thresholds' no seu main():
    calculate_detection_stats(df_boxplot, floor_thresholds)
if __name__ == "__main__":
    main()
