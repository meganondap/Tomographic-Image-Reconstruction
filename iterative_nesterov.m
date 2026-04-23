%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Megan Ondap, Feb 26, 2026
% Parallel-beam tomography on CT images using AIR Tools & Iterative Methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
%% Load CT image
imagePath = fullfile('C:\Users\megan\Box\MeganOndap\MATLAB Code\Images06_png\000308_03_01\053.png');  % change to your file
img = imread(imagePath);

img = im2double(img);
if ndims(img) == 3
img = rgb2gray(img); % Convert to grayscale
end
img = imresize(img, [128 128]); % Ensure it's desired size
N = size(img, 1);
x_true = img(:); % Vectorize for paralleltomo

%% Build AIR Tools system matrix
theta = 0:1:179; % 180 views
p = round(sqrt(2)*N); % Number of rays per view
d = p - 1;
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);

%% Simulate measurement data with noise
b = A_sys * x_true;
noise = 0; % Noise as a percentage
b_noise = b + (noise/100) * max(b) * randn(size(b)); % Add noise to b

%% Initialize Iteration Parameters
max_iter = 150000;
tol = 1e-8;
alpha = 0; % Use Best Alpha
AtA = A_sys' * A_sys;
AtA_reg = AtA + alpha * speye(size(AtA)); % With regularization (alpha)
Atb = A_sys' * b_noise;
sigma_max = svds(A_sys,1,'largest'); % Sigma Max of A
lambda = (1/((sigma_max^2) + alpha)); % Lambda - step size

x_k = zeros(size(A_sys,2),1); % Storage for x(k), x(k+1),...
% Look-head variable for gradient computation
y_k = zeros(size(A_sys,2),1); % Storage for y(k), y(k+1),...
% Acceleration parameter
t_k = 1;

atol = tol*norm(Atb);
%% Iterative Methods to solve matrices
for k = 1:max_iter
    % Evaluate the gradient at y_k (look head)
    gradient_f = A_sys' * (A_sys * y_k - b_noise) + alpha * y_k;
    % Update x_k1
    x_k1 = y_k - lambda * gradient_f;
    % Check for convergence
    if norm(gradient_f) < atol
        fprintf('Converged at iteration %d\n', k);
        break;
    end
    % Compute the acceleration parameter
    t_k1 = 0.5*(1+sqrt(1+4*t_k*t_k));
    beta_k = (t_k - 1)/t_k1;
    % Update the look ahead variable y_k
    y_k = x_k1 + beta_k*(x_k1-x_k);
    % Prepare for next iteration (Update x_k, t_k)
    x_k = x_k1;
    t_k = t_k1;
end
% Final reconstructed vector after gradient descent
x_rec_gd = x_k1;

% Normalized root MSE
nRMSE = norm(x_k1-x_true)/norm(x_true);

% View Original Image
figure;
subplot(1,2,1);
imagesc(img)
title('Original Image');
colormap gray
axis image off;

% Visualize the reconstructed image
subplot(1,2,2); imagesc(reshape(x_rec_gd, N, N)); colormap gray; axis image off;
title(sprintf('Nesterov Iterative Solution (alpha = %d, noise = %d)', alpha, noise));