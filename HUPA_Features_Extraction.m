%% HUPA_Features_Extraction.m
% Calculates aggregated AVCA features (P, R, N) + CPP for each HUPA .wav file.
%
% AVCA blocks included:
%   P: Perturbation-Fluctuation
%   R: Regularity
%   N: Nonlinear / Complexity (NDA)
%
% All columns returned by AVCA_features_stat(sFile, 'PRN') are used
% (means and standard deviations, no feature is discarded).
%
% Additional feature:
%   CPP (Cepstral Peak Prominence, mean over frames) from Covarep
%
% Output: "HUPA_voice_features_PRN_CPP.csv"

clear; clc;

%% ============= 1) PATH CONFIGURATION ===================================
% Detect the current folder of this script
currentPath = fileparts(mfilename('fullpath'));

% Define paths relative to this script
% EXPECTED STRUCTURE:
%   /toolboxes/  <- contains hurst, avca, covarep, etc.
%   /data/       <- contains HUPA_db/Pathol and HUPA_db/Normal
toolboxes_dir = fullfile(currentPath, 'toolboxes');
data_dir      = fullfile(currentPath, 'data');

% Toolbox subfolders (Assumes user extracted them exactly like this)
hurst_root   = fullfile(toolboxes_dir, 'hurst estimators');
me_root      = fullfile(toolboxes_dir, 'ME-master');
rpde_root    = fullfile(toolboxes_dir, 'rpde');
avca_root    = fullfile(toolboxes_dir, 'AVCA-ByO-master');
covarep_root = fullfile(toolboxes_dir, 'covarep-master');
fastdfa_root = fullfile(toolboxes_dir, 'fastdfa');
hctsa_root   = fullfile(toolboxes_dir, 'hctsa-main');

% Database paths
path_pathol  = fullfile(data_dir, 'HUPA_db', 'Pathol');
path_normal  = fullfile(data_dir, 'HUPA_db', 'Normal');

% Output CSV
out_csv      = fullfile(data_dir, 'HUPA_voice_features_PRN_CPP.csv');

%% ============= 2) TOOLBOX LOADING ======================================

restoredefaultpath;

toolboxes = { ...
    hurst_root, ...
    me_root, ...
    rpde_root, ...
    avca_root, ...
    covarep_root, ...
    fastdfa_root, ...
    hctsa_root ...
};

for i = 1:numel(toolboxes)
    root = toolboxes{i};
    if ~exist(root, 'dir')
        warning('Toolbox directory not found: %s', root);
    else
        % Put all custom toolboxes at the beginning of the path
        addpath(genpath(root), '-begin');
    end
end

% Remove old Covarep compatibility folder to avoid conflicts with audioread
backcompat = fullfile(covarep_root, 'external', 'backcompatibility_2015');
if exist(backcompat, 'dir')
    rmpath(backcompat);
end

rehash;
fprintf('Toolboxes loaded and path refreshed.\n');

%% ============= 3) FILE LISTING =========================================

filesPathol = dir(fullfile(path_pathol, '*.wav'));
filesNormal = dir(fullfile(path_normal, '*.wav'));

nPathol = numel(filesPathol);
nNormal = numel(filesNormal);
nTotal  = nPathol + nNormal;

if nTotal == 0
    error('No .wav files found in the specified directories.');
end

fprintf('Found: %d Pathological + %d Normal = %d Total.\n', ...
    nPathol, nNormal, nTotal);

%% ============= 4) GET AVCA FEATURE NAMES (PRN) =========================

% Use one example file to get AVCA names (P+R+N)
if nPathol > 0
    exampleFile = fullfile(path_pathol, filesPathol(1).name);
else
    exampleFile = fullfile(path_normal, filesNormal(1).name);
end

[vec_example, namesPRN] = AVCA_features_stat(exampleFile, 'PRN');

if isempty(vec_example) || isempty(namesPRN)
    error('AVCA_features_stat(exampleFile, ''PRN'') returned empty results.');
end

nFeatAVCA = numel(namesPRN);

% Final feature names = all AVCA PRN names + CPP
featureNamesOut = [namesPRN(:); {'CPP'}];
nFeatOut = numel(featureNamesOut);

fprintf('AVCA PRN features: %d (will add CPP -> total %d columns).\n', ...
    nFeatAVCA, nFeatOut);

%% ============= 5) OUTPUT PRE-ALLOCATION ================================

features  = nan(nTotal, nFeatOut);
fileNames = cell(nTotal, 1);
labels    = nan(nTotal, 1);
global_idx = 1;

%% ============= 6) PROCESS PATHOLOGICAL FILES ===========================

fprintf('\n--- Processing PATHOLOGICAL files ---\n');
for k = 1:nPathol
    sFile = fullfile(path_pathol, filesPathol(k).name);
    fprintf('[%d/%d] %s ... ', k, nPathol, filesPathol(k).name);
    try
        f = extract_features_for_file_PRN_CPP(sFile);
        features(global_idx, :) = f;
        fileNames{global_idx}   = filesPathol(k).name;
        labels(global_idx)      = 1; % Pathological
        fprintf('OK\n');
    catch ME
        fprintf('CRITICAL ERROR: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for kStack = 1:length(ME.stack)
            fprintf('  > In %s (line %d)\n', ...
                ME.stack(kStack).name, ME.stack(kStack).line);
        end
        % leave this row as NaN (label stays NaN)
    end
    global_idx = global_idx + 1;
end

%% ============= 7) PROCESS NORMAL FILES =================================

fprintf('\n--- Processing NORMAL files ---\n');
for k = 1:nNormal
    sFile = fullfile(path_normal, filesNormal(k).name);
    fprintf('[%d/%d] %s ... ', k, nNormal, filesNormal(k).name);
    try
        f = extract_features_for_file_PRN_CPP(sFile);
        features(global_idx, :) = f;
        fileNames{global_idx}   = filesNormal(k).name;
        labels(global_idx)      = 0; % Normal
        fprintf('OK\n');
    catch ME
        fprintf('CRITICAL ERROR: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for kStack = 1:length(ME.stack)
            fprintf('  > In %s (line %d)\n', ...
                ME.stack(kStack).name, ME.stack(kStack).line);
        end
        % leave this row as NaN (label stays NaN)
    end
    global_idx = global_idx + 1;
end

%% ============= 8) SAVE TO CSV ==========================================

% Filter out rows with failed labels
validRows = ~isnan(labels);
features  = features(validRows, :);
fileNames = fileNames(validRows);
labels    = labels(validRows);

T = array2table(features, 'VariableNames', featureNamesOut(:)');
T.FileName = fileNames(:);
T.Label    = labels(:);

writetable(T, out_csv);
fprintf('\nPROCESS COMPLETE. Data saved to: %s\n', out_csv);

%% ============= 9) LOCAL HELPER FUNCTIONS ===============================

function f = extract_features_for_file_PRN_CPP(sFile)
    % Extracts all AVCA PRN features (as returned by AVCA_features_stat)
    % and appends CPP from Covarep.

    % 1) AVCA PRN features (P + R + N, all columns: mean & std)
    [vPRN, ~] = AVCA_features_stat(sFile, 'PRN');
    vPRN = vPRN(:).';  % ensure row vector

    % 2) CPP from Covarep (mean over frames)
    CPP_val = compute_cpp_covarep(sFile);

    % 3) Final feature vector = [all AVCA PRN , CPP]
    f = [vPRN, CPP_val];
end

function cpp_mean = compute_cpp_covarep(sFile)
    % Computes mean Cepstral Peak Prominence (dB) using Covarep.

    [x, fs] = audioread(sFile);

    % Convert to mono if needed
    if size(x,2) > 1
        x = mean(x,2);
    end
    x = x(:);

    % Remove DC and normalize
    x = x - mean(x);
    maxAbs = max(abs(x));
    if maxAbs > 0
        x = x / maxAbs;
    end

    % CPP parameters: smoothing=1, normalization='line', dB scale=1
    CPP = cpp(x, fs, 1, 'line', 1);  % [N x 2], col 1 = CPP, col 2 = time

    vals = CPP(:,1);
    vals = vals(~isnan(vals) & ~isinf(vals));

    if isempty(vals)
        cpp_mean = NaN;
    else
        cpp_mean = mean(vals);
    end
end
