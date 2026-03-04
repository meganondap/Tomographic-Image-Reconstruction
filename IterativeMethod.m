%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Megan Ondap, Jan 28, 2026
% Parallel-beam tomography on CT images using AIR Tools & Iterative Methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load CT image
img = imread('056.png');      % Replace with desired image filename
img = im2double(img);
if ndims(img) == 3
    img = rgb2gray(img);               % Convert to grayscale 
end
img = imresize(img, [64 64]);          % Ensure it's desired size: 40x40, 64x64
N = size(img, 1);                      
x_true = img(:);                       % Vectorize for paralleltomo
%% View Original Image
figure;
imagesc(img)
title('Original Image');
colormap gray
axis image;
%% Build AIR Tools system matrix
theta = 0:1:179;                       % 180 views
p = round(sqrt(2)*N);                  % Number of rays per view
d = p - 1;
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);
%% Simulate measurement data with noise
b = A_sys * x_true;                  
noise = 0;   % Noise as a percentage
b_noise = b + (noise/100) * max(b) * randn(size(b));  % Add noise to b
%% Initialize  Iteration Parameters 
max_iter = 150000;    
tol = 1e-6;

x_k = zeros(size(A_sys,2),1);           % Storage for x(k), x(k+1),...
    
alpha = 0;     % Use Best Alpha
AtA = A_sys' * A_sys;
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha)
Atb = A_sys' * b_noise;

sigma_max = svds(A_sys,1,'largest');    % Sigma Max of A
lambda = (1/((sigma_max^2) + alpha));     % Lambda - step size
%% Iterative Methods to solve matrices

for k = 1:max_iter
    gradient_f = A_sys' * (A_sys * x_k - b_noise) + alpha * x_k;
    x_k1 = x_k - lambda * gradient_f;
    
    % Check for convergence
    if norm(x_k1 - x_k) < tol
        fprintf('Converged at iteration %d\n', k);
        break;
    end
    
    % Prepare for next iteration
    x_k = x_k1;
end

% Final reconstructed vector after gradient descent
x_rec_gd = x_k1;

% Visualize the reconstructed image 
figure; imagesc(reshape(x_rec_gd, N, N)); colormap gray; axis image;
title(sprintf('Iterative Solution (alpha = %d, noise = %d)', alpha, noise));


%% Graph error to pick best alpha using backslash
alphas = 0:5:50;
relative_error = zeros(size(alphas));

for i = 1:length(alphas)
    AtA_reg = AtA + alphas(i) * speye(size(A_sys,2));
    x_rec_alpha = AtA_reg \ Atb; % Reconstruct for each alpha
    relative_error(i) = norm(x_rec_alpha - x_true)/norm(x_true); % Relative error
end

figure;
semilogx(alphas, relative_error, '-o');
xlabel('\alpha'); ylabel('Error');
title('Error vs Alpha');

%% Graph error to pick best alpha using iterative methods
alphas = 0:50:300;
relative_error = zeros(size(alphas));

for i = 1:length(alphas)
    alpha = alphas(i);
    sigma_max = svds(A_sys,1,'largest');
    lambda = 1/((sigma_max^2) + alpha);

    x_k = zeros(size(A_sys,2),1);
    AtA = A_sys' * A_sys;
    Atb = A_sys' * b_noise;
    max_iter = 1000;
    tol = 1e-6;

    for k = 1:max_iter
        gradient_f = AtA * x_k - Atb + alpha * x_k;
        x_k1 = x_k - lambda * gradient_f;
        if norm(x_k1 - x_k) < tol
            break;
        end
        x_k = x_k1;
    end

    x_rec_gd_alpha = x_k1;
    relative_error(i) = norm(x_rec_gd_alpha - x_true)/norm(x_true);
end

figure;
semilogx(alphas, relative_error, '-o');
xlabel('\alpha'); ylabel('Error');
title('Error vs Alpha (Gradient Descent)');
%% Visualize the sinogram of original image
sinogram = reshape(b,p,length(theta)); 
figure;
subplot(1,2,1);
imagesc(sinogram); colormap(gray); axis xy;
xlabel('Angle index'); ylabel('Ray index');
title(sprintf('Simulated Sinogram from CT Image (alpha = %d)', alpha));

% Visualize the reconstructed image
P = reshape(x_true, N, N);             
subplot(1,2,2)
imagesc(P); colormap gray; axis image;
title('Original Image');

%% Visualize the sinogram of noisy image
sinogram = reshape(b_noise,p,length(theta)); 
figure;
subplot(1,2,1);
imagesc(sinogram); colormap(gray); axis xy;
xlabel('Angle index'); ylabel('Ray index');
title(sprintf('Simulated Sinogram from CT Image (alpha = %d)', alpha));

% Visualize the reconstructed image        
subplot(1,2,2) 
imagesc(reshape(x_rec_gd, N, N)); colormap gray; axis image;
title(sprintf('Iterative Solution (alpha = %d, noise = %d)', alpha, noise));
