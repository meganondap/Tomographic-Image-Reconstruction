%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Megan Ondap, Feb 26, 2026
% Parallel-beam tomography on CT images using AIR Tools & Iterative Methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;
%% ---------------- LOAD CT IMAGE ----------------
img = imread('056.png');              % Replace with desired image filename
img = im2double(img);
if ndims(img) == 3
    img = rgb2gray(img);              % Convert to grayscale
end
img = imresize(img, [128 128]);       % Ensure desired size: 40x40, 64x64, 128x128, etc.
N = size(img, 1);
x_true = img(:);                      % Vectorize for paralleltomo

%% ---------------- VIEW ORIGINAL IMAGE ----------------
figure;
imagesc(img);
title('Original Image');
colormap gray;
axis image; 

%% ---------------- BUILD AIR TOOLS SYSTEM MATRIX ----------------
theta = 0:1:179;                      % 180 views
p     = round(sqrt(2)*N);             % Number of rays per view
d     = p - 1;
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);

%% ---------------- SIMULATE MEASUREMENT DATA WITH NOISE ----------------
b = A_sys * x_true;
noise = 0;                            % Noise as a percentage (0 = noiseless)
b_noise = b + (noise/100) * max(b) * randn(size(b));  % Add Gaussian noise to b

%% ---------------- INITIALIZE ITERATION PARAMETERS ----------------
max_iter = 150000;
tol      = 1e-7;

% Single alpha used in the main run (can be 0 or any value)
alpha = 0;

% Precompute normal equations components
AtA = A_sys' * A_sys;
Atb = A_sys' * b_noise;
AtA_reg = AtA + alpha * speye(size(AtA));   % (A^T A + alpha I)

% Step size based on largest singular value of A_sys
sigma_max = svds(A_sys, 1, 'largest');
lambda    = 1 / ((sigma_max^2) + alpha);    % Step size (lambda)

% Convergence tolerance
atol = tol * norm(Atb);

% Initial guess
x_k = zeros(size(A_sys, 2), 1);             % x^(0)

%% ---------------- ITERATIVE GRADIENT DESCENT SOLVE ----------------
for k = 1:max_iter
    % Gradient of ||A x - b||^2 + alpha ||x||^2
    gradient_f = A_sys' * (A_sys * x_k - b_noise) + alpha * x_k;
    
    % Gradient descent update
    x_k1 = x_k - lambda * gradient_f;
    
    % Check for convergence
    if norm(gradient_f) < atol
        fprintf('Converged at iteration %d\n', k);
        break;
    end
    
    % Prepare for next iteration
    x_k = x_k1;
end

%% ---------------- EVALUATE AND VISUALIZE MAIN SOLUTION ----------------
% Normalized root MSE
nRMSE = norm(x_k1 - x_true) / norm(x_true);

% Visualize the reconstructed image
figure;
imagesc(reshape(x_k1, N, N));
colormap gray; axis image; colorbar;
title(sprintf('Iterative Solution (alpha = %1.2e, noise = %1.2f%%%%, nRMSE = %1.2e)',...
              alpha, noise, nRMSE));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------------- ALPHA SCAN: ERROR VS ALPHA ---------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We now scan over a range of alpha values, re-run the iterative method
% for each alpha, and compute the relative error to pick a good alpha.

alphas = 0:10:50;                      % e.g., 0, 10, 20, 30, 40, 50
relative_error = zeros(size(alphas));

for i = 1:length(alphas)
    alpha_i = alphas(i);
    
    % Step size for this alpha
    lambda_i = 1 / ((sigma_max^2) + alpha_i);
    
    % Reset initial guess for each alpha
    x_k = zeros(size(A_sys, 2), 1);
    
    % Iterative gradient descent for this alpha
    for k = 1:max_iter
        grad = A_sys' * (A_sys * x_k - b_noise) + alpha_i * x_k;
        x_k1 = x_k - lambda_i * grad;
        
        if norm(grad) < atol
            break;
        end
        x_k = x_k1;
    end
    
    % Relative error using x_true
    relative_error(i) = norm(x_k1 - x_true) / norm(x_true);
end

%% ---------------- PLOT ERROR VS ALPHA ----------------
figure;
plot(alphas, relative_error, '-o');
xlabel('\alpha');
ylabel('Relative error ||x_\alpha - x_{true}|| / ||x_{true}||');
title(sprintf('Error vs \\alpha (Iterative GD, noise = %1.2f%%%%)', noise));
grid on;

% Report best alpha
[best_err, idx_best] = min(relative_error);
best_alpha = alphas(idx_best);
fprintf('Iterative GD: best alpha = %1.2e with relative error = %1.4e\n',...
        best_alpha, best_err);