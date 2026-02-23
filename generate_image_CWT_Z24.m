%% SCRIPT DE PRÉ-PROCESSAMENTO CWT - Z24 BRIDGE (COM OVERLAP 25%)
% Autor: Adaptado para MATLAB
% Descrição: Processa arquivos .mat da Z24, aplica janelamento com sobreposição,
% CWT e salva como imagens para Deep Learning.

clear; clc; close all;

%% --- Configurações ---
input_folder = '/MATLAB Drive/Doutorado/Z24/Z24_Raw_Data';
output_folder = 'Z24_Imagens_CWT_v1_MATLAB';

% Parâmetros do Sinal e Janelamento
Fs = 100;               % Frequência de amostragem
NumSensors = 7;         % Quantidade de sensores

WindowSize = 1000;      % Tamanho da janela (amostras)
OverlapPerc = 0.25;     % 25% de sobreposição

% Cálculo do Passo (Stride)
% Se Overlap é 25%, andamos 75% da janela para frente
OverlapSamples = floor(WindowSize * OverlapPerc);
StepSize = WindowSize - OverlapSamples; 

% Lista de Arquivos e Cenários
file_list = {
    'Z24_d0.mat', 'SPECIAL_SPLIT'; 
    'Z24_d1.mat', 'd_1';
    'Z24_d2.mat', 'd_2';
    'Z24_d3.mat', 'd_3';
    'Z24_d4.mat', 'd_4';
    'Z24_d5.mat', 'd_5'
};

%% --- Início do Processamento ---
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

disp(['Iniciando processamento com ' num2str(OverlapPerc*100) '% de overlap...']);

for f = 1:size(file_list, 1)
    filename = file_list{f, 1};
    default_scenario = file_list{f, 2};
    
    full_path = fullfile(input_folder, filename);
    
    if ~exist(full_path, 'file')
        warning('Arquivo não encontrado: %s', filename);
        continue;
    end
    
    disp(['Lendo arquivo: ' filename '...']);
    
    % --- Carregamento Dinâmico dos Dados ---
    loaded_struct = load(full_path);
    field_names = fieldnames(loaded_struct);
    data_matrix = [];
    
    for k = 1:length(field_names)
        var_content = loaded_struct.(field_names{k});
        if isnumeric(var_content) && size(var_content, 1) > 1000
            data_matrix = var_content;
            break;
        end
    end
    
    if isempty(data_matrix)
        warning('Dados não encontrados dentro de %s', filename);
        continue;
    end
    
    if size(data_matrix, 2) > NumSensors
        data_matrix = data_matrix(:, 1:NumSensors);
    end
    
    % --- Cálculo de Janelas com Overlap ---
    num_samples = size(data_matrix, 1);
    
    % Fórmula para número de janelas deslizantes: floor((N - W) / S) + 1
    if num_samples >= WindowSize
        num_windows = floor((num_samples - WindowSize) / StepSize) + 1;
    else
        num_windows = 0;
        warning('O arquivo %s é menor que o tamanho da janela!', filename);
    end
    
    % --- Lógica de Split (Intact vs Unknown) ---
    is_split_mode = strcmp(filename, 'Z24_d0.mat');
    split_idx = num_windows; 
    
    if is_split_mode
        split_idx = floor(num_windows * 0.7); 
        fprintf('   -> Modo Split: %d janelas Intact / %d janelas Unknown (Total: %d)\n', split_idx, num_windows - split_idx, num_windows);
    end
    
    % --- Loop por Sensor ---
    for sensor_idx = 1:NumSensors
        sensor_name = sprintf('Sensor%d', sensor_idx);
        signal_full = data_matrix(:, sensor_idx);
        
        % --- Loop por Janela (Deslizante) ---
        for w = 1:num_windows
            % Define índices com base no StepSize
            idx_start = (w-1) * StepSize + 1;
            idx_end = idx_start + WindowSize - 1;
            
            signal_window = signal_full(idx_start:idx_end);
            
            % Define a pasta de destino
            if is_split_mode
                if w <= split_idx
                    target_scenario = 'd_0_intact';
                else
                    target_scenario = 'd_0_unknown';
                end
            else
                target_scenario = default_scenario;
            end
            
            % Cria diretório
            final_dir = fullfile(output_folder, target_scenario, sensor_name);
            if ~exist(final_dir, 'dir')
                mkdir(final_dir);
            end
            
            % --- Processamento CWT e Imagem ---
            [cfs, ~] = cwt(signal_window, 'morse', Fs); 
            
            cfs_abs = abs(cfs);
            cfs_db = cfs_abs;
            
            min_val = min(cfs_db(:));
            max_val = max(cfs_db(:));
            
            if max_val > min_val
                img_norm = (cfs_db - min_val) / (max_val - min_val);
            else
                img_norm = zeros(size(cfs_db));
            end
            
            img_resized = imresize(img_norm, [256 256]);
            img_final = im2uint8(img_resized);
            
            % Salvar Imagem
            [~, file_base_name, ~] = fileparts(filename);
            img_name = sprintf('%s_Win%03d.png', file_base_name, w);
            imwrite(img_final, fullfile(final_dir, img_name));
            
        end % fim loop windows
    end % fim loop sensors
    
    disp(['   -> Concluído: ' filename]);
    
end % fim loop files

disp(' ');
disp('Processamento com Overlap concluído!');