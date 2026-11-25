import os
import glob
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from pytorch_msssim import ssim
from sklearn.model_selection import StratifiedKFold
import gc
import optuna

# ==============================================================================
# CONFIGURATION HEADER AND INPUTS
# ==============================================================================
# --- Directory Settings ---
BASE_DIR = r"C:\Users\Victor Higino\Documents\Doutorado\CCAE\Imagens_CWT_Z24_v1_matlab"
INTACT_CONDITION_FOLDER = 'd_0_intact'
BASE_OUTPUT_DIR = r"C:\Users\Victor Higino\Documents\Doutorado\CCAE\optuna_optimization_output_Z24"

# --- Dataset settings ---
SENSORES = [f'Sensor{i}' for i in range(1, 8)]  # 7 sensores para Z24

# --- List of remaining Classification Weights (Lambda_class) to run sequentially ---
# O peso 1e-1 (0.1) já foi executado; restam 1e-3 (0.001) e 1e-2 (0.01)
CLASSIFICATION_WEIGHTS_TO_TEST = [5e-2]

# --- Fixed Model Settings ---
FIXED_PARAMS = {
    'classifier_linear_dim': 64,
    'bottleneck_channels': 4,
    'reconstruction_loss_weight': 1.0,
    'epoch_warmup': 50,
}

# --- Training Settings and K-Fold ---
N_SPLITS = 4
EPOCHS = 200
EARLY_STOP_PATIENCE = 30
MAX_TRAINING_TIME_PER_FOLD = 1 * 3600

# --- Optuna settings ---
N_TRIALS = 200
BASE_STUDY_NAME = "ccae_hyperparam_optimization_Z24"
BASE_DB_PREFIX = "ccae_optimization_Z24"

# ==============================================================================
# 1. DEFINITION OF THE DATASET (WITH PRE-LOADING AND PRE-TRANSFORMATIONS)
# ==============================================================================
class CWTPreloadedDataset(Dataset):
    def __init__(self, root_folder, sensor_names):
        self.image_data = []
        self.labels = []
        self.sensor_to_idx = {sensor: i for i, sensor in enumerate(sensor_names)}
        
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        print("⏳ Pré-carregando e pré-processando imagens na memória RAM...")
        
        image_paths = []
        for sensor_name in sensor_names:
            sensor_folder = os.path.join(root_folder, '**', sensor_name)
            image_paths.extend(glob.glob(os.path.join(sensor_folder, '**', '*.png'), recursive=True))

        if not image_paths:
            raise ValueError(f"Nenhuma imagem PNG encontrada em: {root_folder} para os sensores especificados.")

        for img_path in image_paths:
            sensor_name = os.path.basename(os.path.dirname(img_path))
            sensor_idx = self.sensor_to_idx.get(sensor_name)
            
            if sensor_idx is not None:
                try:
                    img = Image.open(img_path)
                    tensor_img = self.transform(img)
                    self.image_data.append(tensor_img)
                    self.labels.append(sensor_idx)
                except Exception as e:
                    print(f"❌ Erro ao carregar/transformar imagem {img_path}: {e}. Ignorando.")

        self.labels = np.array(self.labels)
        print(f"✅ Pré-carregamento e pré-processamento concluídos. {len(self.image_data)} imagens carregadas como tensores.")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_tensor = self.image_data[idx]
        label = self.labels[idx]
        return img_tensor, torch.tensor(label, dtype=torch.long)

def collate_fn_stratified(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    images, sensor_ids = zip(*batch)
    return torch.stack(images), torch.stack(sensor_ids)

# ==============================================================================
# 2. LOSS FUNCTION AND MODEL ARCHITECTURE
# ==============================================================================
def ssim_loss(x, y):
    return torch.sqrt(torch.mean((1 - ssim(x, y, data_range=1.0, size_average=False)) ** 2))

classification_loss_fn = nn.CrossEntropyLoss()

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
# 3. OBJECTIVE FUNCTION FOR OPTUNA (PARAMETERIZED BY CURRENT WEIGHT)
# ==============================================================================
def objective(trial, full_dataset, current_class_weight, current_output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trial_dir = os.path.join(current_output_dir, f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)

    # --- 1. Define the hyperparameters to be optimized ---
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    }
    print(f"\n--- [Class Weight = {current_class_weight}] Iniciando Trial {trial.number} com parâmetros: {params} ---")

    labels_for_stratification = full_dataset.labels
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_losses = []

    # --- 2. Cross-Validation Loop (Stratified K-Fold) ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_for_stratification)), labels_for_stratification)):
        print(f"\n--- [Class Weight = {current_class_weight}] Trial {trial.number}, Fold {fold+1}/{N_SPLITS} ---")
        
        train_set = Subset(full_dataset, train_idx)
        val_set = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn_stratified, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn_stratified, pin_memory=True)
        
        model = CCAE(
            num_sensors=len(SENSORES),
            bottleneck_channels=FIXED_PARAMS['bottleneck_channels'],
            classifier_linear_dim=FIXED_PARAMS['classifier_linear_dim'],
            dropout_rate=params['dropout_rate']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=15, min_lr=1e-6)

        best_fold_loss = float('inf')
        best_epoch = -1
        fold_model_path = os.path.join(trial_dir, f'best_model_fold_{fold+1}.pth')
        start_time_fold = time.time()

        for epoch in range(EPOCHS):
            model.train()
            train_total_loss = 0.0
            for images, sensor_ids in train_loader:
                if images.numel() == 0:
                    continue
                images, sensor_ids = images.to(device), sensor_ids.to(device)
                
                optimizer.zero_grad()
                output, sensor_logits = model(images)
                
                recon_loss = ssim_loss(output, images)
                class_loss = classification_loss_fn(sensor_logits, sensor_ids)
                total_loss = (FIXED_PARAMS['reconstruction_loss_weight'] * recon_loss) + \
                             (current_class_weight * class_loss)
                
                total_loss.backward()
                optimizer.step()
                train_total_loss += total_loss.item()
            
            avg_train_total_loss = train_total_loss / len(train_loader) if len(train_loader) > 0 else 0

            model.eval()
            val_total_loss = 0.0
            with torch.no_grad():
                for images, sensor_ids in val_loader:
                    if images.numel() == 0:
                        continue
                    images, sensor_ids = images.to(device), sensor_ids.to(device)
                    output, sensor_logits = model(images)
                    recon_loss = ssim_loss(output, images)
                    class_loss = classification_loss_fn(sensor_logits, sensor_ids)
                    total_loss = (FIXED_PARAMS['reconstruction_loss_weight'] * recon_loss) + \
                                 (current_class_weight * class_loss)
                    val_total_loss += total_loss.item()
            
            avg_val_total_loss = val_total_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            if epoch > FIXED_PARAMS['epoch_warmup']:
                scheduler.step(avg_val_total_loss)

            if avg_val_total_loss < best_fold_loss - 1e-5 and avg_val_total_loss <= avg_train_total_loss * 1.10:
                best_fold_loss = avg_val_total_loss
                best_epoch = epoch
                torch.save(model.state_dict(), fold_model_path)
            
            if epoch - best_epoch >= EARLY_STOP_PATIENCE and best_epoch != -1 and epoch > FIXED_PARAMS['epoch_warmup']:
                break
            
            if time.time() - start_time_fold > MAX_TRAINING_TIME_PER_FOLD:
                break

        fold_losses.append(best_fold_loss)
        
        # --- 3. GPU Memory Release ---
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 4. Pruning ---
        intermediate_value = np.mean(fold_losses)
        trial.report(intermediate_value, fold)
        if trial.should_prune():
            print(f"✂️ Trial {trial.number} podado no fold {fold+1}.")
            raise optuna.exceptions.TrialPruned()
    
    # --- 5. Return the final target value for the trial ---
    final_mean_loss = np.mean(fold_losses)
    print(f"🏁 [Class Weight = {current_class_weight}] Trial {trial.number} concluído. Média das perdas de validação: {final_mean_loss:.5f}")
    return final_mean_loss


# ==============================================================================
# 4. MAIN EXECUTION OF THE SEQUENTIAL OPTUNA STUDIES
# ==============================================================================
if __name__ == "__main__":
    print("=== Iniciando Otimização de Hiperparâmetros (CCAE - Z24) para os 2 Pesos Restantes ===")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Pré-carrega o dataset uma única vez na RAM para todas as execuções
    full_data_path = os.path.join(BASE_DIR, INTACT_CONDITION_FOLDER)
    full_dataset = CWTPreloadedDataset(full_data_path, SENSORES)

    summary_results = {}

    for idx, class_weight in enumerate(CLASSIFICATION_WEIGHTS_TO_TEST, start=1):
        weight_tag = f"{class_weight:.0e}".replace("+", "")
        current_output_dir = os.path.join(BASE_OUTPUT_DIR, f"weight_{weight_tag}")
        os.makedirs(current_output_dir, exist_ok=True)

        current_study_name = f"{BASE_STUDY_NAME}_w_{weight_tag}"
        current_db_filename = f"{BASE_DB_PREFIX}_w_{weight_tag}.db"
        db_path = os.path.join(current_output_dir, current_db_filename)
        storage_url = f"sqlite:///{db_path}"

        print("\n" + "=" * 80)
        print(f"🚀 EXECUÇÃO {idx}/{len(CLASSIFICATION_WEIGHTS_TO_TEST)} | CLASSIFICATION WEIGHT = {class_weight} ({weight_tag})")
        print(f"📂 Diretório de saída: {current_output_dir}")
        print("=" * 80)

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=0,
            interval_steps=1
        )

        study = optuna.create_study(
            study_name=current_study_name,
            storage=storage_url,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True
        )

        try:
            study.optimize(
                lambda trial: objective(trial, full_dataset, class_weight, current_output_dir),
                n_trials=N_TRIALS
            )
        except KeyboardInterrupt:
            print(f"🛑 Otimização para o peso {class_weight} interrompida pelo usuário.")

        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print(f"\n--- Fim da Execução {idx}/{len(CLASSIFICATION_WEIGHTS_TO_TEST)} (Peso = {class_weight}) ---")
        print(f"Nome do Estudo: {study.study_name}")
        print(f"Número de trials finalizados: {len(complete_trials)}")
        print(f"Número de trials podados: {len(pruned_trials)}")

        if complete_trials:
            best_trial = study.best_trial
            summary_results[class_weight] = {
                'trial_number': best_trial.number,
                'best_value': best_trial.value,
                'best_params': best_trial.params
            }
            print("\n--- Melhor Trial para este Peso ---")
            print(f"  Número: {best_trial.number}")
            print(f"  Valor (Média de Loss): {best_trial.value:.5f}")
            print("  Melhores Hiperparâmetros:")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
        else:
            summary_results[class_weight] = None
            print("Nenhum trial foi completado com sucesso para este peso.")

    # --- RESUMO FINAL ---
    print("\n" + "=" * 80)
    print("🏆 RESUMO COMPARATIVO FINAL (CCAE - Z24)")
    print("=" * 80)
    for class_weight, res in summary_results.items():
        if res is not None:
            print(f"Class Weight = {class_weight:.0e} | Melhor Trial: #{res['trial_number']} | Total Loss: {res['best_value']:.5f} | Params: {res['best_params']}")
        else:
            print(f"Class Weight = {class_weight:.0e} | Nenhum trial completado.")