%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CreateTrainingData.m
% Build (X_true, Y) PNG pairs for FNO training
%  - X_true: resized CT image
%  - Y: Nesterov-regularized reconstruction from b = H X_true
%
% Uses AIRToolsII paralleltomo to build system matrix A_sys (H)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

%% ---------------- PARAMETERS ----------------
N = 64;                         % image size N x N (64 or 128)
rootFolder = 'Images06_png';    % root folder with subfolders of CT images
maxSamples = 100;               % maximum number of image pairs to generate

% Output folders for PNGs
outXFolder = 'X_true_png';
outYFolder = 'Y_png';

% Create folders if they don't exist
if ~exist(outXFolder, 'dir'); mkdir(outXFolder); end
if ~exist(outYFolder, 'dir'); mkdir(outYFolder); end

% Clear old files in output folders
delete(fullfile(outXFolder, '*'));
delete(fullfile(outYFolder, '*'));

%% ------------- BUILD SYSTEM MATRIX (H) -------------
% Parallel-beam geometry parameters
theta = 0:0.5:179;              % projection angles (degrees)
p     = round(sqrt(2)*N);       % number of rays per projection
d     = p - 1;

% Build parallel-beam tomography matrix A_sys using AIRToolsII
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);

% Nesterov hyperparameters (fixed for all images)
alpha    = 0;                   % Tikhonov regularization parameter
sigma_max = svds(A_sys, 1, 'largest'); % largest singular value of A_sys
lambda   = 1 / ((sigma_max^2) + alpha); % step size for gradient descent
beta     = 0.9;                 % Nesterov momentum parameter
max_iter = 2000;                % max iterations per image
tol      = 1e-6;                % convergence tolerance

%% ------------- LOOP OVER IMAGES -------------
rootDir = dir(rootFolder);
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
        fprintf('  Image: %s\n', imagename);

        %% --- READ, GRAYSCALE, RESIZE (X_true) ---
        img_orig = imread(imagename);   % original CT slice (uint8/uint16/RGB)

        % Convert to grayscale while still in integer type
        if ndims(img_orig) == 3
            img_gray = rgb2gray(img_orig);
        else
            img_gray = img_orig;
        end

        % Resize to N x N while still integer (cheaper in memory)
        img_gray = imresize(img_gray, [N N]);

        % Convert small image to double in [0,1]
        img = im2double(img_gray);

        % Store as X_true
        X_true = img;                   % N x N double in [0,1]

        %% --- SIMULATE MEASUREMENT DATA b = A_sys * x_true ---
        x_vec = X_true(:);              % vectorize image (N^2 x 1)
        b     = A_sys * x_vec;          % forward projection

        % (Optional) add noise to measurements
        noise   = 0;                    % noise level in percent
        b_noise = b + (noise/100) * max(b) * randn(size(b));

        %% --- NESTEROV ACCELERATED GRADIENT DESCENT TO GET Y ---
        % Solve min_x 0.5||A_sys x - b_noise||^2 + 0.5 alpha ||x||^2

        % Rewrite in python to train neural network

        % Initialize x_0 and x_{-1}
        % x_k   = zeros(size(A_sys,2),1);  % x_0
        % x_km1 = x_k;                     % x_{-1} = x_0

        % for k = 1:max_iter
        %     % Look-ahead point y_k = x_k + beta (x_k - x_{k-1})
        %     y_k = x_k + beta * (x_k - x_km1);
        % 
        %     % Gradient at y_k: grad f(y) = A^T(Ay - b_noise) + alpha * y
        %     grad_f = A_sys' * (A_sys * y_k - b_noise) + alpha * y_k;
        % 
        %     % Nesterov update x_{k+1} = y_k - lambda * grad f(y_k)
        %     x_k1 = y_k - lambda * grad_f;
        % 
        %     % Convergence check
        %     if norm(x_k1 - x_k) < tol
        %         fprintf('    Nesterov converged at iter %d\n', k);
        %         break;
        %     end
        % 
        %     % Shift iterates
        %     x_km1 = x_k;
        %     x_k   = x_k1;
        % end

        % Reshape final iterate into image Y
        y_vec = A_sys' * A_sys * x_vec;
        Y_img = reshape(y_vec, [N N]);     % N x N

        % % Normalize Y to [0,1] for saving
        % Y_norm = Y_img;
        % if max(Y_norm(:)) > 0
        %     Y_norm = Y_norm / max(Y_norm(:));
        % end

        Y_min = min(Y_img(:));
        Y_max = max(Y_img(:));
        
        % Shift and scale to [0,1] using a fixed global range (example)
        % You must decide what global range makes sense physically.
        global_min = 0;      % e.g., 0 if A^T A x_true is nonnegative
        global_max = 1.0;    % or some known upper bound
        
        Y_scaled = (Y_img - global_min) / (global_max - global_min);
        Y_scaled = max(min(Y_scaled, 1), 0);  % clip to [0,1]
        %% --- SAVE PAIR AS PNG ---
        counter = counter + 1;
        idxStr = sprintf('%05d', counter);    % e.g., '00001'

        xFile = fullfile(outXFolder, ['X_true_' idxStr '.png']);
        yFile = fullfile(outYFolder, ['Y_'      idxStr '.png']);

        imwrite(X_true, xFile);   % save resized ground-truth image
        imwrite(Y_scaled, yFile);   % save Nesterov reconstruction

        fprintf('  Saved %s and %s\n', xFile, yFile);

        % Free temporary variables to reduce memory usage
        clear img_orig img_gray img X_true x_vec b b_noise x_k x_km1 x_k1 Y_img Y_norm;

        % Stop after reaching maxSamples
        if counter >= maxSamples
            fprintf('Reached maxSamples = %d. Stopping.\n', maxSamples);
            return;
        end
    end
end

fprintf('Total samples created: %d\n', counter);