%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% backslash.m
%
% Direct (least-squares) reconstruction using MATLAB backslash:
%   x_rec = A_sys \ b
%
% - Builds A_sys
% - Simulates data b = A_sys * x_true
% - Solves for x_rec
% - Displays original and reconstructed images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% ---------------- USER PARAMETERS ----------------
N = 64;  % image size N x N

% Path to a single CT image
imagePath = fullfile('C:\Users\megan\Box\MeganOndap\MATLAB Code\Images06_png\000308_03_01\053.png');  % change to your file

theta = 0:0.5:179;
p     = round(sqrt(2)*N);
d     = p - 1;

%% ---------------- LOAD AND PREPARE IMAGE ----------------
img_orig = imread(imagePath);

if ndims(img_orig) == 3
    img_gray = rgb2gray(img_orig);
else
    img_gray = img_orig;
end

img_gray = imresize(img_gray, [N N]);
X_true = im2double(img_gray);

figure;
subplot(1,2,1);
imagesc(X_true); axis image off; colormap gray;
title('X\_true');

%% ---------------- BUILD A_sys AND SIMULATE DATA ----------------
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);

x_true_vec = X_true(:);
b = A_sys * x_true_vec;  % simulated sinogram
noise = 1; % Noise as a percentage (set to 0 for noiseless)
b_noise = b + (noise/100) * max(b) * randn(size(b)); % Add Gaussian noise

%% ---------------- DIRECT RECONSTRUCTION ----------------
% Solve least-squares problem using backslash
x_rec_vec = A_sys \ b;
X_rec = reshape(x_rec_vec, [N N]);

subplot(1,2,2);
imagesc(X_rec); axis image off; colormap gray;
title('Backslash reconstruction');
sgtitle('Direct reconstruction with MATLAB backslash');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIONAL: Graph error vs alpha to pick best alpha
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
do_alpha_scan = true;  % set false to skip

if do_alpha_scan
    fprintf('\n=== Alpha scan for Tikhonov regularization (backslash) ===\n');
    
    % Precompute normal equations
    AtA = A_sys' * A_sys;
    Atb = A_sys' * b_noise;
    
    % User-defined alpha range
    alphas = 0:50:1000;          % e.g., 0, 10, 20, 30, 40, 50
    relative_error = zeros(size(alphas));
    
    for i = 1:length(alphas)
        alpha = alphas(i);
        AtA_reg = AtA + alpha * speye(size(A_sys,2));
        x_rec_alpha = AtA_reg \ Atb;
        relative_error(i) = norm(x_rec_alpha - x_true_vec) / norm(x_true_vec);
        % fprintf('alpha = %1.2e, rel. error = %1.4e\n', alpha, relative_error(i));
    end
    
    figure;
    plot(alphas, relative_error, '-o');
    xlabel('\alpha');
    ylabel('Relative error ||x_\alpha - x_{true}|| / ||x_{true}||');
    title(sprintf('Error vs \\alpha (Backslash, noise = %1.2f%%)', noise));
    grid on;
    
    % Optional: report best alpha
    [best_err, idx_best] = min(relative_error);
    best_alpha = alphas(idx_best);
    fprintf('Backslash: best alpha = %1.2e with relative error = %1.4e\n', best_alpha, best_err);
end