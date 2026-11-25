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
from sklearn.model_selection import KFold
import gc
import optuna

# ==============================================================================
# CONFIGURATION HEADER AND INPUTS
# ==============================================================================
# --- Directory Settings ---
BASE_DIR = r"C:\Users\Victor Higino\Documents\Doutorado\CCAE\Imagens_CWT_Z24_v1_matlab"
INTACT_CONDITION_FOLDER = 'd_0_intact'
BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    r"C:\Users\Victor Higino\Documents\Doutorado\CCAE\optuna_optimization_output_Z24_SAE_L1"
)

# --- Dataset Settings ---
# Even without using sensor labels, we need to list the folders to load the images
SENSORS = [f'Sensor{i}' for i in range(1, 8)]

# --- List of 3 Sparsity Weights (Lambda) to run sequentially ---
SPARSITY_WEIGHTS_TO_TEST = [1e-3]

# --- Fixed Model Settings ---
FIXED_PARAMS = {
    'bottleneck_channels': 4,
    'epoch_warmup': 50,
    'sparsity_mode': 'l1',      # Options: 'l1' (ReLU + L1) or 'kl' (Sigmoid + KL Divergence)
    'sparsity_target': 0.05,    # Target (rho) used only when sparsity_mode == 'kl'
}

# --- Training and K-Fold Settings ---
N_SPLITS = 4
EPOCHS = 200
EARLY_STOP_PATIENCE = 30
MAX_TRAINING_TIME_PER_FOLD = 1 * 3600

# --- Base Optuna Settings ---
N_TRIALS = 200
BASE_STUDY_NAME = "sae_hyperparam_optimization_Z24"
BASE_DB_PREFIX = "sae_optimization_Z24"

# ==============================================================================
# 1. DATASET DEFINITION (WITHOUT SENSOR LABELS)
# ==============================================================================
class CWTPreloadedDataset(Dataset):
    def __init__(self, root_folder, sensor_names):
        self.image_data = []
        
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        print("⏳ Preloading and preprocessing images into RAM...")
        
        image_paths = []
        for sensor_name in sensor_names:
            sensor_folder = os.path.join(root_folder, '**', sensor_name)
            image_paths.extend(glob.glob(os.path.join(sensor_folder, '**', '*.png'), recursive=True))

        if not image_paths:
            raise ValueError(f"No PNG images found in: {root_folder} for the specified sensors.")

        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                tensor_img = self.transform(img)
                self.image_data.append(tensor_img)
            except Exception as e:
                print(f"❌ Error loading/transforming image {img_path}: {e}. Skipping.")

        print(f"✅ Preloading completed. {len(self.image_data)} images loaded as tensors.")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        return self.image_data[idx]

def collate_fn_simple(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([])
    return torch.stack(batch)

# ==============================================================================
# 2. LOSS FUNCTION AND MODEL ARCHITECTURE (CONVOLUTIONAL SAE)
# ==============================================================================
def ssim_loss(x, y):
    return torch.sqrt(torch.mean((1 - ssim(x, y, data_range=1.0, size_average=False)) ** 2))

def sparsity_loss(encoded_features, mode='l1', rho=0.05):
    if mode == 'l1':
        return torch.mean(torch.abs(encoded_features))
    elif mode == 'kl':
        rho_hat = torch.mean(encoded_features, dim=0)
        rho_hat = torch.clamp(rho_hat, min=1e-6, max=1.0 - 1e-6)
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return torch.mean(kl)
    else:
        raise ValueError(f"Invalid sparsity mode: {mode}")

class ConvSAE(nn.Module):
    def __init__(self, bottleneck_channels, sparsity_mode='l1'):
        super(ConvSAE, self).__init__()
        
        # Encoder (Identical to the traditional CAE for fair comparison)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        
        # Sparse Bottleneck: No BatchNorm2d so activations can be driven toward zero
        bottleneck_activation = nn.Sigmoid() if sparsity_mode == 'kl' else nn.ReLU()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            bottleneck_activation
        )
        
        # Decoder (Identical to the traditional CAE)
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
        return x_reconstructed, encoded_features

# ==============================================================================
# 3. OPTUNA `objective` FUNCTION (PARAMETERIZED BY CURRENT WEIGHT)
# ==============================================================================
def objective(trial, full_dataset, current_sparsity_weight, current_output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trial_dir = os.path.join(current_output_dir, f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)

    # --- 1. Define Hyperparameters to Optimize ---
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        #'bottleneck_channels': trial.suggest_int('bottleneck_channels', 2, 8, step=1),
    }
    print(f"\n--- [Sparse Weight = {current_sparsity_weight}] Starting Trial {trial.number} (SAE) with parameters: {params} ---")

    dataset_indices = np.arange(len(full_dataset))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_losses = []

    # --- 2. Cross-Validation Loop (K-Fold) ---
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
        print(f"\n--- [Sparse Weight = {current_sparsity_weight}] Trial {trial.number}, Fold {fold+1}/{N_SPLITS} ---")
        
        train_set = Subset(full_dataset, train_idx)
        val_set = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn_simple, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn_simple, pin_memory=True)
        
        model = ConvSAE(
            bottleneck_channels=FIXED_PARAMS['bottleneck_channels'],
            sparsity_mode=FIXED_PARAMS['sparsity_mode']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=15, min_lr=1e-6)

        best_fold_loss = float('inf')
        best_epoch = -1
        fold_model_path = os.path.join(trial_dir, f'best_model_fold_{fold+1}.pth')
        start_time_fold = time.time()

        for epoch in range(EPOCHS):
            # --- TRAINING ---
            model.train()
            train_recon_loss = 0.0
            train_sparse_loss = 0.0
            train_total_loss = 0.0
            
            for images in train_loader:
                if images.numel() == 0: continue
                images = images.to(device)
                
                optimizer.zero_grad()
                output, encoded_features = model(images)
                
                recon_loss = ssim_loss(output, images)
                sparse_penalty = sparsity_loss(
                    encoded_features,
                    mode=FIXED_PARAMS['sparsity_mode'],
                    rho=FIXED_PARAMS['sparsity_target']
                )
                weighted_sparse = current_sparsity_weight * sparse_penalty
                total_loss = recon_loss + weighted_sparse
                
                total_loss.backward()
                optimizer.step()
                
                train_recon_loss += recon_loss.item()
                train_sparse_loss += sparse_penalty.item()
                train_total_loss += total_loss.item()
            
            n_train = len(train_loader) if len(train_loader) > 0 else 1
            avg_train_recon = train_recon_loss / n_train
            avg_train_sparse_raw = train_sparse_loss / n_train
            avg_train_sparse_weighted = avg_train_sparse_raw * current_sparsity_weight
            avg_train_total_loss = train_total_loss / n_train

            # --- VALIDATION ---
            model.eval()
            val_recon_loss = 0.0
            val_sparse_loss = 0.0
            val_total_loss = 0.0
            
            with torch.no_grad():
                for images in val_loader:
                    if images.numel() == 0: continue
                    images = images.to(device)
                    output, encoded_features = model(images)
                    
                    recon_loss = ssim_loss(output, images)
                    sparse_penalty = sparsity_loss(
                        encoded_features,
                        mode=FIXED_PARAMS['sparsity_mode'],
                        rho=FIXED_PARAMS['sparsity_target']
                    )
                    weighted_sparse = current_sparsity_weight * sparse_penalty
                    total_loss = recon_loss + weighted_sparse
                    
                    val_recon_loss += recon_loss.item()
                    val_sparse_loss += sparse_penalty.item()
                    val_total_loss += total_loss.item()
            
            n_val = len(val_loader) if len(val_loader) > 0 else 1
            avg_val_recon = val_recon_loss / n_val
            avg_val_sparse_raw = val_sparse_loss / n_val
            avg_val_sparse_weighted = avg_val_sparse_raw * current_sparsity_weight
            avg_val_total_loss = val_total_loss / n_val

            # --- EPOCH METRICS PRINTOUT ---
            #print(
            #    f"Epoch [{epoch+1:03d}/{EPOCHS:03d}] "
            #    f"| TRAIN -> Recon: {avg_train_recon:.5f} | Sparse(w): {avg_train_sparse_weighted:.5f} (raw: {avg_train_sparse_raw:.5f}) | Total: {avg_train_total_loss:.5f} "
            #    f"|| VAL -> Recon: {avg_val_recon:.5f} | Sparse(w): {avg_val_sparse_weighted:.5f} (raw: {avg_val_sparse_raw:.5f}) | Total: {avg_val_total_loss:.5f}"
            #)
            
            if epoch > FIXED_PARAMS['epoch_warmup']: 
                scheduler.step(avg_val_total_loss)

            # Saving criterion: Best Loss and overfitting check (10% margin)
            if avg_val_total_loss < best_fold_loss - 1e-5 and avg_val_total_loss <= avg_train_total_loss * 1.10:
                best_fold_loss = avg_val_total_loss
                best_epoch = epoch
                torch.save(model.state_dict(), fold_model_path)
            
            if epoch - best_epoch >= EARLY_STOP_PATIENCE and best_epoch != -1 and epoch > FIXED_PARAMS['epoch_warmup']:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1} (Best epoch: {best_epoch+1}).")
                break
            
            if time.time() - start_time_fold > MAX_TRAINING_TIME_PER_FOLD:
                print(f"⏰ Fold time limit reached at epoch {epoch+1}.")
                break

        fold_losses.append(best_fold_loss)
        
        # --- 3. Memory Cleanup ---
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 4. Pruning ---
        intermediate_value = np.mean(fold_losses)
        trial.report(intermediate_value, fold)
        if trial.should_prune():
            print(f"✂️ Trial {trial.number} pruned at fold {fold+1}.")
            raise optuna.exceptions.TrialPruned()
    
    # --- 5. Final Return ---
    final_mean_loss = np.mean(fold_losses)
    print(f"🏁 [Sparse Weight = {current_sparsity_weight}] Trial {trial.number} completed. Mean validation loss: {final_mean_loss:.5f}")
    return final_mean_loss


# ==============================================================================
# 4. MAIN EXECUTION OF THE 3 SEQUENTIAL OPTUNA STUDIES
# ==============================================================================
if __name__ == "__main__":
    print("=== Starting Hyperparameter Optimization Suite (Convolutional SAE) for 3 Weights ===")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Preload dataset once into RAM for all 3 runs
    full_data_path = os.path.join(BASE_DIR, INTACT_CONDITION_FOLDER)
    full_dataset = CWTPreloadedDataset(full_data_path, SENSORS)

    summary_results = {}

    for idx, sparsity_weight in enumerate(SPARSITY_WEIGHTS_TO_TEST, start=1):
        weight_tag = f"{sparsity_weight:.0e}".replace("+", "")
        current_output_dir = os.path.join(BASE_OUTPUT_DIR, f"weight_{weight_tag}")
        os.makedirs(current_output_dir, exist_ok=True)

        current_study_name = f"{BASE_STUDY_NAME}_w_{weight_tag}"
        current_db_filename = f"{BASE_DB_PREFIX}_w_{weight_tag}.db"
        db_path = os.path.join(current_output_dir, current_db_filename)
        storage_url = f"sqlite:///{db_path}"

        print("\n" + "=" * 80)
        print(f"🚀 RUN {idx}/{len(SPARSITY_WEIGHTS_TO_TEST)} | SPARSITY WEIGHT (LAMBDA) = {sparsity_weight} ({weight_tag})")
        print(f"📂 Output directory: {current_output_dir}")
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
                lambda trial: objective(trial, full_dataset, sparsity_weight, current_output_dir),
                n_trials=N_TRIALS
            )
        except KeyboardInterrupt:
            print(f"🛑 Optimization for weight {sparsity_weight} interrupted by user.")

        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print(f"\n--- End of Run {idx}/{len(SPARSITY_WEIGHTS_TO_TEST)} (Sparsity Weight = {sparsity_weight}) ---")
        print(f"Study Name: {study.study_name}")
        print(f"Number of completed trials: {len(complete_trials)}")
        print(f"Number of pruned trials: {len(pruned_trials)}")

        if complete_trials:
            best_trial = study.best_trial
            summary_results[sparsity_weight] = {
                'trial_number': best_trial.number,
                'best_value': best_trial.value,
                'best_params': best_trial.params
            }
            print("\n--- Best Trial for this Weight ---")
            print(f"  Trial Number: {best_trial.number}")
            print(f"  Value (Mean Loss): {best_trial.value:.5f}")
            print("  Best Hyperparameters:")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
        else:
            summary_results[sparsity_weight] = None
            print("No trials were completed successfully for this weight.")

    # --- FINAL SUMMARY ACROSS ALL 3 RUNS ---
    print("\n" + "=" * 80)
    print("🏆 FINAL COMPARATIVE SUMMARY (ALL 3 WEIGHTS - SAE)")
    print("=" * 80)
    for sparsity_weight, res in summary_results.items():
        if res is not None:
            print(f"Sparse Weight = {sparsity_weight:.0e} | Best Trial: #{res['trial_number']} | Total Loss: {res['best_value']:.5f} | Params: {res['best_params']}")
        else:
            print(f"Sparse Weight = {sparsity_weight:.0e} | No completed trials.")