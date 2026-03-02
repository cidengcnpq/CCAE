import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import optuna
from sklearn.decomposition import PCA
import pandas as pd
# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================
BASE_DIR_OUTPUT = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo"
OUTPUT_DIR = os.path.join(BASE_DIR_OUTPUT, 'optuna_optimization_output_yellow_frame')
STUDY_NAME = "ccae_hyperparam_optimization_yellow_frame"
DB_FILENAME = "ccae_optimization_yellow_frame.db"

ROOT_DATA_DIR = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo/imagens_cwt_yellow_frame"

FIXED_PARAMS = {
    'classifier_linear_dim': 64,
    'bottleneck_channels': 4,
}

ALL_SENSORS_MODEL_TRAINED_ON = [f'Sensor{i}' for i in range(1, 16)]
NUM_SENSORS_IN_TRAINED_MODEL = len(ALL_SENSORS_MODEL_TRAINED_ON)
DAMAGE_SCENARIOS = ['d_0_intact', 'd_0_unknown', 'd_1', 'd_2']
INTACT_CONDITION_NAME = 'd_0_intact'

# ==============================================================================
#1. FUNCTION TO LOAD THE BEST OPTUNA STUDY MODEL
# ==============================================================================
def find_best_model_from_study(study_name, storage_url, output_dir):
    """Loads an Optuna study, finds the best trial, and returns the path to the first-fold model and its hyperparameters.."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        best_trial = study.best_trial
        print("--- Melhor Trial Encontrado ---")
        print(f"  Número do Trial: {best_trial.number}")
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
#2. DEFINITION OF THE CCAE MODEL
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

# ==============================================================================
# 3. DEFINITIONS OF DATASET AND DATALOADER
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
            sensor_name = os.path.basename(os.path.dirname(img_path))
            ground_truth_idx = self.sensor_to_idx.get(sensor_name)
            return self.transform(img), ground_truth_idx, os.path.basename(img_path)
        except Exception as e:
            print(f"Erro ao carregar imagem {img_path}: {e}. Retornando None.")
            return None, None, None

def collate_fn_analysis(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return torch.tensor([]), [], []
    images, ground_truths, names = zip(*batch)
    return torch.stack(images), list(ground_truths), list(names)

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

def load_best_model(model_path, device, num_sensors, fixed_params, best_hyperparams):
    """Instantiate the model with the correct parameters and load the weights."""
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
#4. FUNCTION FOR VIEWING SENSOR EMBEDDINGS
# ==============================================================================
def plot_sensor_embeddings(model, all_sensor_data):
    """
    Plots the pure sensor embeddings and the mixed embeddings from each sample 
on a grid of plots, using PCA. Markers indicate correct ('o') and 
incorrect ('x') model classifications.
    """
    print("\n--- Gerando visualização PCA dos embeddings: Análise de Classificação ---")

    pure_sensor_embeddings = model.sensor_embedding.weight.data.cpu().numpy()
    num_sensors = pure_sensor_embeddings.shape[0]
    sensor_names = [f'$Sensor_{{{i}}}$' for i in range(1, num_sensors + 1)]

    mixed_data_by_scenario = {}
    all_mixed_embeddings_list = []

    print("Extraindo embeddings e classificações de todas as amostras...")
    model.eval()
    with torch.no_grad():
        for sensor_name, conditions in all_sensor_data.items():
            for condition, dataloader in conditions.items():
                if dataloader:
                    if condition not in mixed_data_by_scenario:
                        mixed_data_by_scenario[condition] = {'embeddings': [], 'correct': []}
                    
                    for images, ground_truths, _ in dataloader:
                        if images.numel() == 0: continue
                        images = images.to(model.sensor_embedding.weight.device)
                        
                        x = model.encoder(images)
                        encoded_features = model.bottleneck_conv(x)
                        sensor_logits = model.classifier(encoded_features)
                        
                        predicted_classes = torch.argmax(sensor_logits, dim=1).cpu().numpy()
                        ground_truths_array = np.array(ground_truths)
                        is_correct = (predicted_classes == ground_truths_array)

                        sensor_probabilities = torch.softmax(sensor_logits, dim=1)
                        all_sensor_embeddings = model.sensor_embedding(torch.arange(model.num_sensors).to(images.device))
                        combined_sensor_emb = (sensor_probabilities.unsqueeze(2) * all_sensor_embeddings.unsqueeze(0)).sum(dim=1)
                        
                        mixed_data_by_scenario[condition]['embeddings'].extend(combined_sensor_emb.cpu().numpy())
                        mixed_data_by_scenario[condition]['correct'].extend(is_correct)
    
    for scenario in DAMAGE_SCENARIOS:
        if scenario in mixed_data_by_scenario:
            all_mixed_embeddings_list.extend(mixed_data_by_scenario[scenario]['embeddings'])

    all_mixed_embeddings = np.array(all_mixed_embeddings_list)
    all_embeddings = np.vstack([pure_sensor_embeddings, all_mixed_embeddings])
    
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(all_embeddings)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    pc1_variance = explained_variance_ratio[0] * 100
    pc2_variance = explained_variance_ratio[1] * 100
    xlabel_text = f"Principal Component 1 ({pc1_variance:.1f}% expl. var.)"
    ylabel_text = f"Principal Component 2 ({pc2_variance:.1f}% expl. var.)"
 
    pure_pca = embeddings_pca[:num_sensors, :]
    
    mixed_pca_dict = {}
    start_idx = num_sensors
    for scenario in DAMAGE_SCENARIOS:
        if scenario in mixed_data_by_scenario:
            num_points = len(mixed_data_by_scenario[scenario]['embeddings'])
            end_idx = start_idx + num_points
            mixed_pca_dict[scenario] = {
                'embeddings': embeddings_pca[start_idx:end_idx, :],
                'correct': np.array(mixed_data_by_scenario[scenario]['correct'])
            }
            start_idx = end_idx

    # ==========================================================================
    # PLOTTING
    # ==========================================================================
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'
    
    # Increase the font size for better readability on the larger plot
    plt.rcParams.update({'font.size': 30})

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(26, 26), sharex=True, sharey=True)
    ax_flat = axes.flatten()
    
    sns.set_style("whitegrid")
    
    scenario_colors = {
        'd_0_intact': '#4285F4',
        'd_0_unknown': '#34A853',
        'd_1': '#EA4335',
        'd_2': '#FBBC05',
    }
    
    subplot_titles = ['(a) Intact (training/validation)', '(b) Intact (testing)', '(c) Damage 1', '(d) Damage 2']

    for idx, scenario in enumerate(DAMAGE_SCENARIOS):
        ax = ax_flat[idx]
        color = scenario_colors[scenario]
        
        if scenario in mixed_pca_dict:
            embeddings_pca_data = mixed_pca_dict[scenario]['embeddings']
            is_correct = mixed_pca_dict[scenario]['correct']
            
            correct_points = embeddings_pca_data[is_correct]
            incorrect_points = embeddings_pca_data[~is_correct]

            ax.scatter(correct_points[:, 0], correct_points[:, 1],
                       c=color, s=60, alpha=0.7, zorder=1, marker='o', label='Correct')
            
            ax.scatter(incorrect_points[:, 0], incorrect_points[:, 1],
                       c=color, s=80, alpha=0.9, zorder=1, marker='x', linewidths=3.0, label='Incorrect')
        
        ax.scatter(pure_pca[:, 0], pure_pca[:, 1], c='none', s=350, marker='D',
                   edgecolors='black', linewidths=3.0, zorder=2, label='Pure Embedding')
        
        for i, txt in enumerate(sensor_names):
            offset_y = -35 if i == 0 else -18
            ax.annotate(txt, (pure_pca[i, 0], pure_pca[i, 1]),
                        xytext=(18, offset_y), textcoords='offset points', fontsize=22, zorder=3)
        
        ax.set_title(subplot_titles[idx], fontsize=40, fontweight='bold', loc='center')
        ax.set_xlabel(xlabel_text, fontsize=38)
        ax.set_ylabel(ylabel_text, fontsize=38)
        
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    legend_handles = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='none', markeredgecolor='black', markersize=18, label='Pure Embedding', linewidth=0, markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='gray', markersize=18, label='Hybrid Embedding: Correct Class', linewidth=0),
        plt.Line2D([0], [0], marker='x', color='gray', markersize=18, label='Hybrid Embedding: Incorrect Class', linewidth=0, markeredgewidth=3)
    ]
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.05),
               ncol=3, fontsize=31, frameon=True, fancybox=True, shadow=True)
    
    save_dir = os.path.dirname(OUTPUT_DIR)
    
    plot_filename_png = os.path.join(save_dir, "embeddings_pca_grid_scatter_plot_classification_large.png")
    fig.savefig(plot_filename_png, dpi=600, bbox_inches='tight')
    print(f"\n✅ Plot PNG de alta qualidade e maior salvo em: {plot_filename_png}")
    
    plot_filename_svg = os.path.join(save_dir, "embeddings_pca_grid_scatter_plot_classification_large.svg")
    fig.savefig(plot_filename_svg, format='svg', bbox_inches='tight')
    print(f"✅ Plot SVG (vetorizado) maior salvo em: {plot_filename_svg}")
    
    plt.show()
    
def calcular_tabela_erros(model, all_sensor_data, device):
    """
    It calculates the classification error rate for each scenario and generates the article sentence.
    """
    print("\n--- Calculando Taxas de Erro de Classificação por Cenário ---")
    
    # Dictionary for accumulating totals by scenario
    # Structure: { 'scenario_name': {'total': 0, 'errors': 0} }
    stats = {}

    model.eval()
    with torch.no_grad():
        # Iterates over sensors.
        for sensor_name, conditions in all_sensor_data.items():
            # Iterates over conditions (scenarios)
            for condition, dataloader in conditions.items():
                if dataloader is None:
                    continue
                
                if condition not in stats:
                    stats[condition] = {'total': 0, 'erros': 0}

                for images, ground_truths, _ in dataloader:
                    if images.numel() == 0: continue
                    
                    images = images.to(device)
                    # Ground truths
                    labels = torch.tensor(ground_truths).to(device)

                    # Inference (only the part necessary for classification)
                    x = model.encoder(images)
                    encoded_features = model.bottleneck_conv(x)
                    sensor_logits = model.classifier(encoded_features)
                    
                    # Prediction
                    _, predicted = torch.max(sensor_logits, 1)
                    
                    # Accounting
                    batch_total = labels.size(0)
                    batch_errors = (predicted != labels).sum().item()
                    
                    stats[condition]['total'] += batch_total
                    stats[condition]['erros'] += batch_errors

    # Table creation
    table_data = []
    for condition, data in stats.items():
        if data['total'] > 0:
            error_rate = (data['erros'] / data['total']) * 100
            accuracy = 100 - error_rate
            table_data.append({
                'Cenário': condition,
                'Amostras Totais': data['total'],
                'Erros': data['erros'],
                'Taxa de Erro (%)': f"{error_rate:.2f}%",
                'Acurácia (%)': f"{accuracy:.2f}%"
            })
    
    df = pd.DataFrame(table_data)
    
    order_map = {v: k for k, v in enumerate(['d_0_intact', 'd_0_unknown', 'd_1', 'd_2', 'd_3', 'd_4', 'd_5'])}
    df['sort_key'] = df['Cenário'].map(order_map)
    df = df.sort_values('sort_key').drop('sort_key', axis=1)

    print("\nResumo de Classificação:")
    print(df.to_string(index=False))
    
# ==============================================================================
# 5. MAIN FUNCTION
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

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

    sensor_groups_by_floor = {
        '1_andar': ['Sensor1', 'Sensor2', 'Sensor3'],
        '2_andar': ['Sensor4', 'Sensor5', 'Sensor6'],
        '3_andar': ['Sensor7', 'Sensor8', 'Sensor9'],
        '4_andar': ['Sensor10', 'Sensor11', 'Sensor12'],
        '5_andar': ['Sensor13', 'Sensor14', 'Sensor15']
    }
    all_sensors_for_analysis = [s for floor_sensors in sensor_groups_by_floor.values() for s in floor_sensors]

    print("\n--- Carregando dados para o gráfico ---")
    all_sensor_data = load_all_sensor_data(ROOT_DATA_DIR, DAMAGE_SCENARIOS, all_sensors_for_analysis, ALL_SENSORS_MODEL_TRAINED_ON)
    
    plot_sensor_embeddings(model, all_sensor_data)

    calcular_tabela_erros(model, all_sensor_data, device)
if __name__ == "__main__":

    main()
