import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import optuna
import random
from pytorch_msssim import ssim 

# ==============================================================================
# SETTINGS
# ==============================================================================

BASE_DIR_OUTPUT = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo"
OUTPUT_DIR = os.path.join(BASE_DIR_OUTPUT, 'optuna_optimization_output_Z24')
STUDY_NAME = "ccae_hyperparam_optimization_Z24"
DB_FILENAME = "ccae_optimization_Z24.db"

# --- Data Analysis Directory Settings ---
ROOT_DATA_DIR = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo/imagens_CWT_Z24_v1_matlab"

FIXED_PARAMS = {
    'classifier_linear_dim': 64,
    'bottleneck_channels': 4,
}
NUM_SENSORS = 7

# --- TEST CONFIGURATION ---
TARGET_SENSOR = 'Sensor3'       # Original sensor
INCORRECT_SENSOR_IDX = 6         # "Liar" Sensor (Index 6 = Sensor 7)
SAMPLE_IDX = random.randint(0, 500)  # Image index to retrieve

# ==============================================================================
# MODEL DEFINITION
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
        x_enc = self.encoder(x)
        encoded_features = self.bottleneck_conv(x_enc)
        sensor_logits = self.classifier(encoded_features)
        sensor_probabilities = torch.softmax(sensor_logits, dim=1)
        
        all_sensor_embeddings = self.sensor_embedding(torch.arange(self.num_sensors).to(x.device))
        combined_sensor_emb = (sensor_probabilities.unsqueeze(2) * all_sensor_embeddings.unsqueeze(0)).sum(dim=1)
        
        combined_sensor_emb_reshaped = combined_sensor_emb.view(-1, 1, 8, 8)
        concatenated_features = torch.cat([encoded_features, combined_sensor_emb_reshaped], dim=1)
        x_reconstructed = self.decoder(concatenated_features)
        return x_reconstructed

    def forward_forced_context(self, x, forced_sensor_idx):
        x_enc = self.encoder(x)
        encoded_features = self.bottleneck_conv(x_enc)
        
        batch_size = x.size(0)
        forced_probs = torch.zeros(batch_size, self.num_sensors).to(x.device)
        forced_probs[:, forced_sensor_idx] = 1.0
        
        all_sensor_embeddings = self.sensor_embedding(torch.arange(self.num_sensors).to(x.device))
        combined_sensor_emb = (forced_probs.unsqueeze(2) * all_sensor_embeddings.unsqueeze(0)).sum(dim=1)
        
        combined_sensor_emb_reshaped = combined_sensor_emb.view(-1, 1, 8, 8)
        concatenated_features = torch.cat([encoded_features, combined_sensor_emb_reshaped], dim=1)
        x_reconstructed = self.decoder(concatenated_features)
        
        return x_reconstructed

# ==============================================================================
# Auxiliary Functions
# ==============================================================================

def load_best_model_optuna(study_name, base_output_dir):
    db_path = os.path.join(base_output_dir, DB_FILENAME)
    storage_url = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        best_trial = study.best_trial
        print(f"Melhor trial carregado: {best_trial.number}")
        model_path = os.path.join(base_output_dir, f'trial_{best_trial.number}', 'best_model_fold_1.pth')
        return model_path, best_trial.params
    except Exception as e:
        print(f"Erro Optuna: {e}")
        return None, None

def get_sample_image(root_dir, sensor_name, condition='d_0_intact'):
    search_path = os.path.join(root_dir, condition, sensor_name, "*.png")
    files = glob.glob(search_path)
    if not files: raise FileNotFoundError(f"Sem imagens em {search_path}")
    img_path = files[min(SAMPLE_IDX, len(files)-1)]
    img_filename = os.path.basename(img_path)
    print(f"Imagem: {img_filename}")
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path)
    return transform(img).unsqueeze(0), img_filename

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Model
    model_path, best_params = load_best_model_optuna(STUDY_NAME, OUTPUT_DIR)
    if not model_path: return

    model = CCAE(NUM_SENSORS, FIXED_PARAMS['bottleneck_channels'], 
                  FIXED_PARAMS['classifier_linear_dim'], best_params['dropout_rate']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load images
    input_tensor, img_filename = get_sample_image(ROOT_DATA_DIR, TARGET_SENSOR)
    input_tensor = input_tensor.to(device)

    # 3. Generate Reconstructions and Calculate Metrics
    with torch.no_grad():
        rec_standard = model(input_tensor)
        rec_incorrect = model.forward_forced_context(input_tensor, forced_sensor_idx=INCORRECT_SENSOR_IDX)
        
        mse_std = nn.MSELoss()(rec_standard, input_tensor).item()
        mse_wrong = nn.MSELoss()(rec_incorrect, input_tensor).item()
        
        ssim_std = ssim(rec_standard, input_tensor, data_range=1.0, size_average=True).item()
        ssim_wrong = ssim(rec_incorrect, input_tensor, data_range=1.0, size_average=True).item()

    # 4. Prepare for Plotting
    img_orig = input_tensor.squeeze().cpu().numpy()
    img_std = rec_standard.squeeze().cpu().numpy()
    img_wrong = rec_incorrect.squeeze().cpu().numpy()

    res_std = np.abs(img_orig - img_std)
    res_wrong = np.abs(img_orig - img_wrong)

    # --- DEFINITION OF COMMON SCALES ---
    imgs_max_val = np.max([img_orig.max(), img_std.max(), img_wrong.max()])
    imgs_min_val = 0.0 
    err_max_val = np.max([res_std.max(), res_wrong.max()])
    err_min_val = 0.0

    # 5. Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cmap_img = 'gray'
    cmap_err = 'inferno'

    # --- Line 1: Images ---
    
    ax = axes[0, 0]
    im1 = ax.imshow(img_orig, cmap=cmap_img, vmin=imgs_min_val, vmax=imgs_max_val)
    ax.set_title(f"(a) Original Input\nZ24 Bridge: {TARGET_SENSOR} (File: {img_filename})", 
                 fontsize=11, fontfamily='Times new roman', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # Standard Reconstruction
    ax = axes[0, 1]
    im2 = ax.imshow(img_std, cmap=cmap_img, vmin=imgs_min_val, vmax=imgs_max_val)
    ax.set_title(f"(b) Standard Reconstruction\nSSIM: {ssim_std:.4f} | MSE: {mse_std:.5f}", 
                  fontsize=11, fontfamily='Times new roman', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Incorrect Reconstruction
    ax = axes[0, 2]
    im3 = ax.imshow(img_wrong, cmap=cmap_img, vmin=imgs_min_val, vmax=imgs_max_val)
    ax.set_title(f"(c) Forced Sensor {INCORRECT_SENSOR_IDX+1} Context\nSSIM: {ssim_wrong:.4f} | MSE: {mse_wrong:.5f}", 
                  fontsize=11, fontfamily='Times new roman', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, "Absolute Difference Maps\n|Original - Reconstructed|", 
                    ha='center', va='center', fontsize=13, fontfamily='Times new roman', fontweight='bold')

    # Standard Residual
    ax = axes[1, 1]
    im4 = ax.imshow(res_std, cmap=cmap_err, vmin=err_min_val, vmax=err_max_val)
    ax.set_title(f"(d) Residual Standard", fontsize=11, fontfamily='Times new roman', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

    # Incorrect Residual
    ax = axes[1, 2]
    im5 = ax.imshow(res_wrong, cmap=cmap_err, vmin=err_min_val, vmax=err_max_val)
    ax.set_title(f"(e) Residual Incorrect Context", fontsize=11, fontfamily='Times new roman', fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im5, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Pixel Error', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    main()
