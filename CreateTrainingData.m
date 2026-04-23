%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CreateTrainingData.m
% Build (X_true, Z_tilde) pairs for FNO training
%  - X_true: resized CT image, grayscale, [0,1]
%  - Z: A_sys^T A_sys X_true
%  - Z_tilde: globally scaled Z in [0,1]
%
% Uses AIRToolsII paralleltomo to build system matrix A_sys
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

%% ---------------- USER PARAMETERS ----------------
N = 64;                         % image size N x N (64 or 128)
maxSamples = 1000;                % maximum number of image pairs to generate

% === USER-DEFINED DIRECTORIES ===
% Make the folders for the outputs & copy their addresses here

% Root folder with subfolders of CT images
rootFolder = 'C:\Users\megan\Box\MeganOndap\MATLAB Code\Images06_png';

% Folder where PNGs will be saved (X_true and Z_tilde)
outPngFolder = 'C:\Users\megan\Box\MeganOndap\MATLAB Code\PNGs';

% Folder where .MAT files will be saved
outMatFolder = 'C:\Users\megan\Box\MeganOndap\MATLAB Code\MATs';

%% --------- CREATE / CLEAN OUTPUT FOLDERS (If needed) ----------
if ~exist(outPngFolder, 'dir'); mkdir(outPngFolder); end
if ~exist(outMatFolder, 'dir'); mkdir(outMatFolder); end

% Clear old files in output folders
delete(fullfile(outPngFolder, '*.png'));
delete(fullfile(outMatFolder, '*.mat'));

%% ------------- BUILD SYSTEM MATRIX (A_sys) -------------
% Parallel-beam geometry parameters
theta = 0:0.5:179;              % projection angles (degrees)
p     = round(sqrt(2)*N);       % number of rays per projection
d     = p - 1;

% Build parallel-beam tomography matrix A_sys using AIRToolsII
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);

%% ------------- PASS 1: ESTIMATE GLOBAL MIN/MAX OF A_sys^T A_sys X_true -------------
fprintf('=== PASS 1: Estimating global_min and global_max over up to %d samples ===\n', maxSamples);

% Initialization
rootDir    = dir(rootFolder);
counter    = 0;
global_max = -inf;
global_min =  inf;

for i = 1:length(rootDir)
    % Needed because root folder has subfolders for each patient
    if ~rootDir(i).isdir, continue; end
    name = rootDir(i).name;
    if strcmp(name, '.') || strcmp(name, '..'), continue; end

    subFolderPath = fullfile(rootDir(i).folder, name);
    fprintf('Subfolder: %s\n', subFolderPath);

    filesinfolder = dir(subFolderPath);

    for j = 1:length(filesinfolder)
        if filesinfolder(j).isdir, continue; end
        if strcmp(filesinfolder(j).name, '.') || strcmp(filesinfolder(j).name, '..'), continue; end

        imagename = fullfile(filesinfolder(j).folder, filesinfolder(j).name);
        fprintf('  [Pass 1] Image: %s\n', imagename);

        %% --- READ, GRAYSCALE, RESIZE TO N X N (X_true) ---
        img_orig = imread(imagename);   % original CT slice (uint8/uint16/RGB)

        % Convert to grayscale while still in integer type
        if ndims(img_orig) == 3
            img_gray = rgb2gray(img_orig);
        else
            img_gray = img_orig;
        end

        % Resize to N x N while still integer
        img_gray = imresize(img_gray, [N N]);

        % Convert small image to double in [0,1]
        X_true = im2double(img_gray);   % N x N double in [0,1]

        %% --- COMPUTE Y = A_sys^T A_sys X_true ---
        x_vec = X_true(:);                  % vectorize image (N^2 x 1)
        z_vec = A_sys' * A_sys * x_vec;     % Define Z = A_sys^T A_sys x_true
        Z_img = reshape(z_vec, [N N]);      % N x N

        % Update global_min and global_max
        this_max = max(Z_img(:));
        this_min = min(Z_img(:));

        % if this is larger then the global max, then update global max
        if this_max > global_max
            global_max = this_max;
        end
        
        % if this is smaller then the global min, then update global min
        if this_min < global_min
            global_min = this_min;
        end

        % update counter
        counter = counter + 1;

        if counter >= maxSamples
            fprintf('Reached maxSamples = %d in PASS 1. Stopping.\n', maxSamples);
            break;
        end
    end
    
    % troubleshooting breaks
    if counter >= maxSamples
        break;
    end
end

if ~(global_max > global_min)
    error('Invalid global_min/global_max. Check your data or A_sys.');
end

% Success Message
fprintf('PASS 1 complete. Estimated global_min = %.6e, global_max = %.6e over %d samples.\n',...
        global_min, global_max, counter);

%% ------------- PASS 2: GENERATE AND SAVE (X_true, Z, Z_tilde) -------------
fprintf('\n=== PASS 2: Generating training pairs using global_min/global_max ===\n');

% Reset counters
counter = 0;

for i = 1:length(rootDir)
    if ~rootDir(i).isdir, continue; end
    name = rootDir(i).name;
    if strcmp(name, '.') || strcmp(name, '..'), continue; end

    subFolderPath = fullfile(rootDir(i).folder, name);
    fprintf('Subfolder: %s\n', subFolderPath);

    filesinfolder = dir(subFolderPath);

    for j = 1:length(filesinfolder)
        if filesinfolder(j).isdir, continue; end
        if strcmp(filesinfolder(j).name, '.') || strcmp(filesinfolder(j).name, '..'), continue; end

        imagename = fullfile(filesinfolder(j).folder, filesinfolder(j).name);
        fprintf('  [Pass 2] Image: %s\n', imagename);

        %% --- READ, GRAYSCALE, RESIZE (X_true) ---
        img_orig = imread(imagename);   % original CT slice

        % Convert to grayscale while still in integer type
        if ndims(img_orig) == 3
            img_gray = rgb2gray(img_orig);
        else
            img_gray = img_orig;
        end

        % Resize to N x N while still integer
        img_gray = imresize(img_gray, [N N]);

        % Convert small image to double in [0,1]
        X_true = im2double(img_gray);   % N x N double in [0,1]

        %% --- COMPUTE z = A_sys^T A_sys X_true ---
        x_vec = X_true(:);                  % vectorize image (N^2 x 1)
        z_vec = A_sys' * A_sys * x_vec;     % A_sys^T A_sys x_true
        Z_img = reshape(z_vec, [N N]);      % N x N

        %% --- GLOBAL SCALING: Z_tilde = (Z - global_min)/(global_max - global_min) ---
        Z_tilde = (Z_img - global_min) / (global_max - global_min);

        %% --- SAVE AS PNG (OPTIONAL) ---
        counter = counter + 1;
        idxStr = sprintf('%05d', counter);    % e.g., '00001'

        xFilePng = fullfile(outPngFolder, ['X_true_'   idxStr '.png']);
        zFilePng = fullfile(outPngFolder, ['Z_tilde_'  idxStr '.png']);

        imwrite(X_true,  xFilePng);    % X_true in [0,1]
        imwrite(Z_tilde, zFilePng);    % Z_tilde in [0,1]

        %% --- SAVE AS.MAT: X_true, Z (unscaled), Z_tilde ---
        matFile = fullfile(outMatFolder, ['sample_' idxStr '.mat']);
        X_true_mat  = X_true;
        Z_mat       = Z_img;
        Z_tilde_mat = Z_tilde;

        save(matFile, 'X_true_mat', 'Z_mat', 'Z_tilde_mat','global_min', 'global_max');

        fprintf('  Saved PNGs: %s, %s\n', xFilePng, zFilePng);
        fprintf('  Saved MAT:  %s\n', matFile);

        % Free temporary variables
        clear img_orig img_gray X_true x_vec z_vec Z_img Z_tilde...
              X_true_mat Z_mat Z_tilde_mat;

        if counter >= maxSamples
            fprintf('Reached maxSamples = %d in PASS 2. Stopping.\n', maxSamples);
            fprintf('Total samples created: %d\n', counter);
            return;
        end
    end
end

fprintf('Total samples created: %d\n', counter);