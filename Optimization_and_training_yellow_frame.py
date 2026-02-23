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
# CABEÇALHO DE CONFIGURAÇÃO E INPUTS
# ==============================================================================
# --- Configurações de Diretórios ---
BASE_DIR = "C:/Users/Alienware/Documents/Victor_Higino/imagens_cwt_yellow_frame"
INTACT_CONDITION_FOLDER = 'd_0_intact'
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'C:/Users/Alienware/Documents/Victor_Higino/optuna_optimization_output_yellow_frame')

# --- Configurações do Dataset ---
SENSORES = [f'Sensor{i}' for i in range(1, 16)]#15 sensores para yellow frame

# --- Configurações do Modelo Fixo ---
FIXED_PARAMS = {
    'classifier_linear_dim': 64,
    'bottleneck_channels': 4,
    'reconstruction_loss_weight': 1.0,
    'classification_loss_weight': 0.1,
    'epoch_warmup':50,
}

# --- Configurações do Treinamento e K-Fold ---
N_SPLITS = 4
EPOCHS = 200
EARLY_STOP_PATIENCE = 30
MAX_TRAINING_TIME_PER_FOLD = 1 * 3600

# --- Configurações do Optuna ---
N_TRIALS = 200
STUDY_NAME = "ccae_hyperparam_optimization_yellow_frame"
DB_FILENAME = "ccae_optimization_yellow_frame.db"

# ==============================================================================
# 1. DEFINIÇÃO DO DATASET (COM PRÉ-CARREGAMENTO E PRÉ-TRANSFORMAÇÕES)
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
                    # Aplica a transformação imediatamente e armazena o tensor
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
# 2. FUNÇÃO DE PERDA E ARQUITETURA DO MODELO
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
# 3. FUNÇÃO `objective` PARA O OPTUNA
# ==============================================================================
def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trial_dir = os.path.join(OUTPUT_DIR, f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)

    # --- 1. Definir Hiperparâmetros a serem otimizados ---
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    }
    print(f"\n--- Iniciando Trial {trial.number} com parâmetros: {params} ---")

    # --- 2. Preparar Dataset e K-Fold ---
    try:
        full_data_path = os.path.join(BASE_DIR, INTACT_CONDITION_FOLDER)
        # O dataset é criado aqui e todas as transformações fixas já são aplicadas.
        full_dataset = CWTPreloadedDataset(full_data_path, SENSORES)
        labels_for_stratification = full_dataset.labels
    
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Erro fatal ao carregar o dataset: {e}")
        raise optuna.exceptions.TrialPruned()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_losses = []

    # --- 3. Loop de Validação Cruzada (K-Fold) ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels_for_stratification)), labels_for_stratification)):
        print(f"\n--- Trial {trial.number}, Fold {fold+1}/{N_SPLITS} ---")
        
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
                if images.numel() == 0: continue
                images, sensor_ids = images.to(device), sensor_ids.to(device)
                
                optimizer.zero_grad()
                output, sensor_logits = model(images)
                
                recon_loss = ssim_loss(output, images)
                class_loss = classification_loss_fn(sensor_logits, sensor_ids)
                total_loss = (FIXED_PARAMS['reconstruction_loss_weight'] * recon_loss) + \
                             (FIXED_PARAMS['classification_loss_weight'] * class_loss)
                
                total_loss.backward()
                optimizer.step()
                train_total_loss += total_loss.item()
            
            avg_train_total_loss = train_total_loss / len(train_loader) if len(train_loader) > 0 else 0

            model.eval()
            val_total_loss = 0.0
            with torch.no_grad():
                for images, sensor_ids in val_loader:
                    if images.numel() == 0: continue
                    images, sensor_ids = images.to(device), sensor_ids.to(device)
                    output, sensor_logits = model(images)
                    recon_loss = ssim_loss(output, images)
                    class_loss = classification_loss_fn(sensor_logits, sensor_ids)
                    total_loss = (FIXED_PARAMS['reconstruction_loss_weight'] * recon_loss) + \
                                 (FIXED_PARAMS['classification_loss_weight'] * class_loss)
                    val_total_loss += total_loss.item()
            
            avg_val_total_loss = val_total_loss / len(val_loader) if len(val_loader) > 0 else 0
            #print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_total_loss:.5f} | Val Loss: {avg_val_total_loss:.5f}")
            if epoch>FIXED_PARAMS['epoch_warmup']: 
                scheduler.step(avg_val_total_loss)

            if avg_val_total_loss < best_fold_loss - 1e-5 and avg_val_total_loss <= avg_train_total_loss * 1.10: #10%
                best_fold_loss = avg_val_total_loss
                best_epoch = epoch
                torch.save(model.state_dict(), fold_model_path)
                #print(f"✅ Melhor modelo do fold salvo em epoch {epoch+1} com loss {best_fold_loss:.5f}")
            
            if epoch - best_epoch >= EARLY_STOP_PATIENCE and best_epoch != -1 and epoch>FIXED_PARAMS['epoch_warmup']:
                #print(f"⏹️ Early stopping na época {epoch+1}.")
                break
            
            if time.time() - start_time_fold > MAX_TRAINING_TIME_PER_FOLD:
                #print(f"⏳ Tempo máximo de treino para o fold atingido.")
                break

        fold_losses.append(best_fold_loss)
        
        # --- 4. Liberação de Memória da GPU ---
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 5. Pruning (Poda) ---
        # Reporta a média das perdas dos folds concluídos até agora
        intermediate_value = np.mean(fold_losses)
        trial.report(intermediate_value, fold)
        if trial.should_prune():
            print(f"✂️ Trial {trial.number} podado no fold {fold+1}.")
            raise optuna.exceptions.TrialPruned()
    
    # --- 6. Retornar o valor objetivo final para o trial ---
    final_mean_loss = np.mean(fold_losses)
    print(f"🏁 Trial {trial.number} concluído. Média das perdas de validação: {final_mean_loss:.5f}")
    return final_mean_loss


# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL DO ESTUDO OPTUNA
# ==============================================================================
if __name__ == "__main__":
    print("--- Iniciando Otimização de Hiperparâmetros com Optuna ---")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    db_path = os.path.join(OUTPUT_DIR, DB_FILENAME)
    storage_url = f"sqlite:///{db_path}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=0,
        interval_steps=1
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("🛑 Otimização interrompida pelo usuário.")
    
    print("\n--- Otimização Concluída ---")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print(f"Nome do Estudo: {study.study_name}")
    print(f"Número de trials finalizados: {len(complete_trials)}")
    print(f"Número de trials podados: {len(pruned_trials)}")

    if complete_trials:
        best_trial = study.best_trial
        print("\n--- Melhor Trial ---")
        print(f"  Número: {best_trial.number}")
        print(f"  Valor (Média de Loss): {best_trial.value:.5f}")
        print("  Melhores Hiperparâmetros:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:

        print("Nenhum trial foi completado com sucesso.")
