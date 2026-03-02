import optuna
import os
import pandas as pd
import numpy as np

# ==============================================================================
# 1. CONFIGURATION OF THE 4 CASES
# ==============================================================================
BASE_DIR_OUTPUT = "C:/Users/F9S4/OneDrive - PETROBRAS/Área de Trabalho/códigos artigo"

CASES = [
    {
        "id": "Yellow Frame - Proposed CCAE",
        "output_folder": "optuna_optimization_output_yellow_frame",
        "db_filename": "ccae_optimization_yellow_frame.db",
        "study_name": "ccae_hyperparam_optimization_yellow_frame",
        "n_splits": 4 
    },
    {
        "id": "Yellow Frame - Traditional CAE",
        "output_folder": "optuna_optimization_output_yellow_frame_CAE",
        "db_filename": "cae_optimization_yellow_frame.db",
        "study_name": "cae_hyperparam_optimization_yellow_frame",
        "n_splits": 4
    },
    {
        "id": "Z24 Bridge - Proposed CCAE",
        "output_folder": "optuna_optimization_output_Z24",
        "db_filename": "ccae_optimization_Z24.db",
        "study_name": "ccae_hyperparam_optimization_Z24",
        "n_splits": 4
    },
    {
        "id": "Z24 Bridge - Traditional CAE",
        "output_folder": "optuna_optimization_output_Z24_CAE",
        "db_filename": "cae_optimization_Z24.db",
        "study_name": "cae_hyperparam_optimization_Z24",
        "n_splits": 4
    }
]

# ==============================================================================
# 2. PROCESSING FUNCTION
# ==============================================================================
def process_study(case_config):
    db_path = os.path.join(BASE_DIR_OUTPUT, case_config["output_folder"], case_config["db_filename"])
    storage_url = f"sqlite:///{db_path}"
    n_splits = case_config["n_splits"]
    
    print(f"🔄 Processando: {case_config['id']}...")
    
    if not os.path.exists(db_path):
        print(f"   ❌ Arquivo não encontrado: {db_path}")
        return None

    try:
        study = optuna.load_study(study_name=case_config["study_name"], storage=storage_url)
        
        # --- A. DATA FROM THE BEST TRIAL ("The Winner") ---
        best_trial = study.best_trial
        best_loss = best_trial.value
        
        # Specific time of the best trial (in minutes)
        best_trial_duration_min = (best_trial.datetime_complete - best_trial.datetime_start).total_seconds() / 60.0
        
        # --- B. GENERAL STATISTICS (All Trials Completed) ---
        df = study.trials_dataframe()
        df_complete = df[df['state'] == 'COMPLETE'].copy()
        
        if not df_complete.empty:
            # Convert dates
            df_complete['datetime_start'] = pd.to_datetime(df_complete['datetime_start'])
            df_complete['datetime_complete'] = pd.to_datetime(df_complete['datetime_complete'])
            
            # 1. Time per Trial (Total folds)
            df_complete['trial_duration_min'] = (df_complete['datetime_complete'] - df_complete['datetime_start']).dt.total_seconds() / 60.0
            mean_trial = df_complete['trial_duration_min'].mean()
            std_trial = df_complete['trial_duration_min'].std()
            
            # 2. Time per Fold (Estimated: Total / Number of Splits)
            df_complete['fold_duration_min'] = df_complete['trial_duration_min'] / n_splits
            mean_fold = df_complete['fold_duration_min'].mean()
            std_fold = df_complete['fold_duration_min'].std()
            
            count_complete = len(df_complete)
        else:
            mean_trial, std_trial = 0, 0
            mean_fold, std_fold = 0, 0
            count_complete = 0

        # String formatting for the table (Mean ± Standard Deviation)
        trial_str = f"{mean_trial:.2f} ± {std_trial:.2f}"
        fold_str = f"{mean_fold:.2f} ± {std_fold:.2f}"

        return {
            "Case ID": case_config["id"],
            "Best Loss": best_loss,
            "Best Trial Time (min)": round(best_trial_duration_min, 2), # <--- NOVO
            "Trial Time (Mean ± Std)": trial_str,
            "Fold Time (Mean ± Std)": fold_str,
            "N Trials": count_complete
        }

    except Exception as e:
        print(f"   ❌ Erro ao ler estudo: {e}")
        return None

# ==============================================================================
# 3. TABLE GENERATION AND DISPLAY
# ==============================================================================
results_list = []

for case in CASES:
    result = process_study(case)
    if result:
        results_list.append(result)

df_final = pd.DataFrame(results_list)

# Defining the column order for logical display
cols_order = [
    "Case ID", 
    "Best Loss", 
    "Best Trial Time (min)", 
    "Trial Time (Mean ± Std)", 
    "Fold Time (Mean ± Std)",
    "N Trials"
]

print("\n" + "="*120)
print("📊 TABELA COMPLETA PARA O ARTIGO")
print("="*120)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

# Displays the organized table
print(df_final[cols_order])

CASES = [
    {
        "id": "Yellow Frame - Proposed CCAE",
        "folder": "optuna_optimization_output_yellow_frame",
        "db": "ccae_optimization_yellow_frame.db",
        "study": "ccae_hyperparam_optimization_yellow_frame"
    },
    {
        "id": "Yellow Frame - Traditional CAE",
        "folder": "optuna_optimization_output_yellow_frame_CAE",
        "db": "cae_optimization_yellow_frame.db",
        "study": "cae_hyperparam_optimization_yellow_frame"
    },
    {
        "id": "Z24 Bridge - Proposed CCAE",
        "folder": "optuna_optimization_output_Z24",
        "db": "ccae_optimization_Z24.db",
        "study": "ccae_hyperparam_optimization_Z24"
    },
    {
        "id": "Z24 Bridge - Traditional CAE",
        "folder": "optuna_optimization_output_Z24_CAE",
        "db": "cae_optimization_Z24.db",
        "study": "cae_hyperparam_optimization_Z24"
    }
]

results = []

print(f"{'Case ID':<35} | {'LR':<10} | {'Batch':<6} | {'Dropout':<8}")
print("-" * 70)

for case in CASES:
    db_path = os.path.join(BASE_DIR_OUTPUT, case["folder"], case["db"])
    storage_url = f"sqlite:///{db_path}"
    
    try:
        study = optuna.load_study(study_name=case["study"], storage=storage_url)
        params = study.best_params
        
        # Formatting Learning Rate to Scientific Notation
        lr = params.get('learning_rate', 0)
        lr_str = f"{lr:.2e}"
        
        batch = params.get('batch_size', 'N/A')
        dropout = params.get('dropout_rate', 'N/A')
        
        print(f"{case['id']:<35} | {lr_str:<10} | {batch:<6} | {dropout:<8}")
        
        results.append({
            "Case ID": case["id"],
            "Learning Rate": lr_str,
            "Batch Size": batch,
            "Dropout Rate": dropout
        })
        
    except Exception as e:

        print(f"Error reading {case['id']}: {e}")
