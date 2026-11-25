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
BASE_DIR = r"C:\Users\Victor Higino\MATLAB Drive\Doutorado\Yellow_frame\imagens_cwt_yellow_frame"
INTACT_CONDITION_FOLDER = 'd_0_intact'
BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    r"C:\Users\Victor Higino\Documents\Doutorado\CCAE\optuna_optimization_output_yellow_frame_VAE"
)

# --- Dataset Settings ---
# Even without using sensor labels, we need to list the folders to load the images
SENSORS = [f'Sensor{i}' for i in range(1, 16)]

# --- List of 3 Weights (Beta) to run sequentially ---
KL_WEIGHTS_TO_TEST = [1e-3, 1e-2, 1e-1]

# --- Fixed Model Settings ---
FIXED_PARAMS = {
    'bottleneck_channels': 4,
    'epoch_warmup': 50,
}

# --- Training and K-Fold Settings ---
N_SPLITS = 4
EPOCHS = 200
EARLY_STOP_PATIENCE = 30
MAX_TRAINING_TIME_PER_FOLD = 1 * 3600

# --- Base Optuna Settings ---
N_TRIALS = 200
BASE_STUDY_NAME = "vae_hyperparam_optimization_Z24"
BASE_DB_PREFIX = "vae_optimization_Z24"

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
# 2. LOSS FUNCTION AND MODEL ARCHITECTURE (CONVOLUTIONAL VAE)
# ==============================================================================
def ssim_loss(x, y):
    return torch.sqrt(torch.mean((1 - ssim(x, y, data_range=1.0, size_average=False)) ** 2))

def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

class ConvVAE(nn.Module):
    def __init__(self, bottleneck_channels):
        super(ConvVAE, self).__init__()
        
        # Encoder (Identical to the traditional CAE for fair comparison)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        
        # Variational Bottleneck: projects to mean (mu) and log-variance (logvar)
        self.conv_mu = nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1)
        self.conv_logvar = nn.Conv2d(128, bottleneck_channels, kernel_size=3, stride=1, padding=1)
        
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

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        x = self.encoder(x)
        mu = self.conv_mu(x)
        # Clamp prevents exponential overflow (NaN) at high learning rates
        logvar = torch.clamp(self.conv_logvar(x), min=-10.0, max=10.0)
        
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

# ==============================================================================
# 3. OPTUNA `objective` FUNCTION (PARAMETERIZED BY CURRENT WEIGHT)
# ==============================================================================
def objective(trial, full_dataset, current_kl_weight, current_output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trial_dir = os.path.join(current_output_dir, f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)

    # --- 1. Define Hyperparameters to Optimize ---
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        #'bottleneck_channels': trial.suggest_int('bottleneck_channels', 2, 8, step=1),
    }
    print(f"\n--- [KL Weight = {current_kl_weight}] Starting Trial {trial.number} (VAE) with parameters: {params} ---")

    dataset_indices = np.arange(len(full_dataset))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_losses = []

    # --- 2. Cross-Validation Loop (K-Fold) ---
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
        print(f"\n--- [KL Weight = {current_kl_weight}] Trial {trial.number}, Fold {fold+1}/{N_SPLITS} ---")
        
        train_set = Subset(full_dataset, train_idx)
        val_set = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn_simple, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn_simple, pin_memory=True)
        
        model = ConvVAE(
            bottleneck_channels=FIXED_PARAMS['bottleneck_channels']
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
            train_kl_loss = 0.0
            train_total_loss = 0.0
            
            for images in train_loader:
                if images.numel() == 0: continue
                images = images.to(device)
                
                optimizer.zero_grad()
                output, mu, logvar = model(images)
                
                recon_loss = ssim_loss(output, images)
                kl_loss = kl_divergence_loss(mu, logvar)
                weighted_kl = current_kl_weight * kl_loss
                total_loss = recon_loss + weighted_kl
                
                total_loss.backward()
                optimizer.step()
                
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                train_total_loss += total_loss.item()
            
            n_train = len(train_loader) if len(train_loader) > 0 else 1
            avg_train_recon = train_recon_loss / n_train
            avg_train_kl_raw = train_kl_loss / n_train
            avg_train_kl_weighted = avg_train_kl_raw * current_kl_weight
            avg_train_total_loss = train_total_loss / n_train

            # --- VALIDATION ---
            model.eval()
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_total_loss = 0.0
            
            with torch.no_grad():
                for images in val_loader:
                    if images.numel() == 0: continue
                    images = images.to(device)
                    output, mu, logvar = model(images)
                    
                    recon_loss = ssim_loss(output, images)
                    kl_loss = kl_divergence_loss(mu, logvar)
                    weighted_kl = current_kl_weight * kl_loss
                    total_loss = recon_loss + weighted_kl
                    
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    val_total_loss += total_loss.item()
            
            n_val = len(val_loader) if len(val_loader) > 0 else 1
            avg_val_recon = val_recon_loss / n_val
            avg_val_kl_raw = val_kl_loss / n_val
            avg_val_kl_weighted = avg_val_kl_raw * current_kl_weight
            avg_val_total_loss = val_total_loss / n_val

            # --- EPOCH METRICS PRINTOUT ---
            #print(
            #    f"Epoch [{epoch+1:03d}/{EPOCHS:03d}] "
            #    f"| TRAIN -> Recon: {avg_train_recon:.5f} | KL(w): {avg_train_kl_weighted:.5f} (raw: {avg_train_kl_raw:.5f}) | Total: {avg_train_total_loss:.5f} "
            #    f"|| VAL -> Recon: {avg_val_recon:.5f} | KL(w): {avg_val_kl_weighted:.5f} (raw: {avg_val_kl_raw:.5f}) | Total: {avg_val_total_loss:.5f}"
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
    print(f"🏁 [KL Weight = {current_kl_weight}] Trial {trial.number} completed. Mean validation loss: {final_mean_loss:.5f}")
    return final_mean_loss


# ==============================================================================
# 4. MAIN EXECUTION OF THE 3 SEQUENTIAL OPTUNA STUDIES
# ==============================================================================
if __name__ == "__main__":
    print("=== Starting Hyperparameter Optimization Suite (Convolutional VAE) for 3 Weights ===")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Preload dataset once into RAM for all 3 runs
    full_data_path = os.path.join(BASE_DIR, INTACT_CONDITION_FOLDER)
    full_dataset = CWTPreloadedDataset(full_data_path, SENSORS)

    summary_results = {}

    for idx, kl_weight in enumerate(KL_WEIGHTS_TO_TEST, start=1):
        weight_tag = f"{kl_weight:.0e}".replace("+", "")
        current_output_dir = os.path.join(BASE_OUTPUT_DIR, f"weight_{weight_tag}")
        os.makedirs(current_output_dir, exist_ok=True)

        current_study_name = f"{BASE_STUDY_NAME}_w_{weight_tag}"
        current_db_filename = f"{BASE_DB_PREFIX}_w_{weight_tag}.db"
        db_path = os.path.join(current_output_dir, current_db_filename)
        storage_url = f"sqlite:///{db_path}"

        print("\n" + "=" * 80)
        print(f"🚀 RUN {idx}/{len(KL_WEIGHTS_TO_TEST)} | KL WEIGHT (BETA) = {kl_weight} ({weight_tag})")
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
                lambda trial: objective(trial, full_dataset, kl_weight, current_output_dir),
                n_trials=N_TRIALS
            )
        except KeyboardInterrupt:
            print(f"🛑 Optimization for weight {kl_weight} interrupted by user.")

        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print(f"\n--- End of Run {idx}/{len(KL_WEIGHTS_TO_TEST)} (KL Weight = {kl_weight}) ---")
        print(f"Study Name: {study.study_name}")
        print(f"Number of completed trials: {len(complete_trials)}")
        print(f"Number of pruned trials: {len(pruned_trials)}")

        if complete_trials:
            best_trial = study.best_trial
            summary_results[kl_weight] = {
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
            summary_results[kl_weight] = None
            print("No trials were completed successfully for this weight.")

    # --- FINAL SUMMARY ACROSS ALL 3 RUNS ---
    print("\n" + "=" * 80)
    print("🏆 FINAL COMPARATIVE SUMMARY (ALL 3 WEIGHTS - VAE)")
    print("=" * 80)
    for kl_weight, res in summary_results.items():
        if res is not None:
            print(f"KL Weight = {kl_weight:.0e} | Best Trial: #{res['trial_number']} | Total Loss: {res['best_value']:.5f} | Params: {res['best_params']}")
        else:
            print(f"KL Weight = {kl_weight:.0e} | No completed trials.")