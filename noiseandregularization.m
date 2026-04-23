clear; clc;

%% Load CT image
imagePath = fullfile('C:\Users\megan\Box\MeganOndap\MATLAB Code\Images06_png\000308_03_01\053.png');  % change to your file
img = imread(imagePath);

img = im2double(img);
if ndims(img) == 3
    img = rgb2gray(img);               % Convert to grayscale 
end
img = imresize(img, [64 64]);          % Ensure it's 40x40
N = size(img, 1);                      % N = 40
x_true = img(:);                       % Vectorize to 1600 × 1 for paralleltomo

%% Build AIR Tools system matrix
theta = 0:1:179;                       % 180 views
p = round(sqrt(2)*N);                  % Number of rays per view
d = p - 1;
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);

%% Simulate measurement data without noise
b = A_sys * x_true;                    % Simulated sinogram (10,260 × 1)
b_noise = b+0.0000*max(b)*randn(size(b));         % Add noise

% Solve the normal equations using backslash
alpha = 0;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
figure; subplot (3,5,1);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('0% Noise');% Reconstructed vector (1,600 × 1)

% Graph error to pick best alpha
alphas = 0:0.1:1;
relative_error = zeros(size(alphas));

for i = 1:length(alphas)
    AtA_reg = AtA + alphas(i) * speye(size(A_sys,2));
    x_rec = AtA_reg \ Atb;
    relative_error(i) = norm(x_rec - x_true)/norm(x_true); % Relative error, euclidean norm
end

subplot(3,5,6);
plot(alphas, relative_error, '-o');
xlabel('\alpha'); ylabel('Error');

b = A_sys * x_true;                    % Simulated sinogram (10,260 × 1)
b_noise = b+0.0000*max(b)*randn(size(b));         % Add noise

% Solve the normal equations using backslash
alpha = 0;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
subplot (3,5,11);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('\alpha=0');% Reconstructed vector (1,600 × 1)

%% Simulate measurement data with noise
b = A_sys * x_true;                    % Simulated sinogram (10,260 × 1)
b_noise = b+0.00001*max(b)*randn(size(b));         % Add noise

% Solve the normal equations using backslash
alpha = 0;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
subplot (3,5,2);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('0.001% Noise');% Reconstructed vector (1,600 × 1)

% Graph error to pick best alpha
alphas = 0:0.05:0.5;
relative_error = zeros(size(alphas));

for i = 1:length(alphas)
    AtA_reg = AtA + alphas(i) * speye(size(A_sys,2));
    x_rec = AtA_reg \ Atb;
    relative_error(i) = norm(x_rec - x_true)/norm(x_true); % Relative error, euclidean norm
end

subplot(3,5,7);
plot(alphas, relative_error, '-o');
xlabel('\alpha'); ylabel('Error');

% Solve the normal equations using backslash
alpha = 0.05;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
subplot (3,5,12);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('\alpha=0.05');% Reconstructed vector (1,600 × 1)
%% Simulate measurement data with noise
b = A_sys * x_true;                    % Simulated sinogram (10,260 × 1)
b_noise = b+0.0001*max(b)*randn(size(b));         % Add noise

% Solve the normal equations using backslash
alpha = 0;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
subplot (3,5,3);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('0.01% Noise');% Reconstructed vector (1,600 × 1)

% Graph error to pick best alpha
alphas = 0:0.25:3;
relative_error = zeros(size(alphas));

for i = 1:length(alphas)
    AtA_reg = AtA + alphas(i) * speye(size(A_sys,2));
    x_rec = AtA_reg \ Atb;
    relative_error(i) = norm(x_rec - x_true)/norm(x_true); % Relative error, euclidean norm
end

subplot(3,5,8);
plot(alphas, relative_error, '-o');
xlabel('\alpha'); ylabel('Error');

% Solve the normal equations using backslash
alpha = 1.25;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
subplot (3,5,13);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('\alpha=1.25');% Reconstructed vector (1,600 × 1)

%% Simulate measurement data with noise
b = A_sys * x_true;                    % Simulated sinogram (10,260 × 1)
b_noise = b+0.001*max(b)*randn(size(b));         % Add noise

% Solve the normal equations using backslash
alpha = 0;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
subplot (3,5,4);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('0.1% Noise');% Reconstructed vector (1,600 × 1)

% Graph error to pick best alpha
alphas = 0:12.5:100;
relative_error = zeros(size(alphas));

for i = 1:length(alphas)
    AtA_reg = AtA + alphas(i) * speye(size(A_sys,2));
    x_rec = AtA_reg \ Atb;
    relative_error(i) = norm(x_rec - x_true)/norm(x_true); % Relative error, euclidean norm
end

subplot(3,5,9);
plot(alphas, relative_error, '-o');
xlabel('\alpha'); ylabel('Error');

% Solve the normal equations using backslash
alpha = 25;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
subplot (3,5,14);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('\alpha=25');% Reconstructed vector (1,600 × 1)

%% Simulate measurement data with noise
b = A_sys * x_true;                    % Simulated sinogram (10,260 × 1)
b_noise = b+0.01*max(b)*randn(size(b));         % Add noise

% Solve the normal equations using backslash
alpha = 0;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
subplot (3,5,5);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('1% Noise');% Reconstructed vector (1,600 × 1)


% Graph error to pick best alpha
alphas = 0:100:1000;
relative_error = zeros(size(alphas));

for i = 1:length(alphas)
    AtA_reg = AtA + alphas(i) * speye(size(A_sys,2));
    x_rec = AtA_reg \ Atb;
    relative_error(i) = norm(x_rec - x_true)/norm(x_true); % Relative error, euclidean norm
end

subplot(3,5,10);
plot(alphas, relative_error, '-o');
xlabel('\alpha'); ylabel('Error');

% Solve the normal equations using backslash
alpha = 200;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
subplot (3,5,15);imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('\alpha=200');% Reconstructed vector (1,600 × 1)
