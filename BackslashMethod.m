%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Megan Ondap, Jan 28, 2026
%
% Description of file
%   Parallel tomo on the CT 40x40 image
%   Image Reconstruction using backslash operations
%   Options to pick best alpha & view sinograms
%
% Description of inputs/outputs
%   Inputs:
%       N = number of discretization intervals, consists of N^2 cells
%       theta = vector containing projection angles in degrees. 
%           Default: theta = 0:1:179
%       p = # of rays. 
%           Default: p = round(sqrt(2)*N
%       d = scalar denoting distance from first ray to the last
%           Default: d = p-1
%       isDisp - If isDisp is non-zero it specifies the time in seconds
%             to pause in the display of the rays. If zero (the default),
%             no display is shown.
%       isMatrix - If non-zero, a sparse matrix is returned in A (default).
%             If zero, instead a function handle is returned.
%   Outputs:
%       A = coefficient matrix with N^2 columns & length(theta)*p rows
%       b = vector containing the rhs of the test problem
%       x = vector containg exact solution, with elements between 0 & 1
%       theta = vector containing the used angles in degrees
%       p = # of rays used for each angle
%       d = distance between first & last ray

%% Load CT image
img = imread('CTImage40x40.png');      % Replace with desired image filename
img = im2double(img);
if ndims(img) == 3
    img = rgb2gray(img);               % Convert to grayscale 
end
img = imresize(img, [40 40]);          % Ensure it's 40x40
N = size(img, 1);                      % N = 40
x_true = img(:);                       % Vectorize to 1600 × 1 for paralleltomo
%% View Image
imagesc(img)
colormap gray
axis equal
%% Build AIR Tools system matrix
theta = 0:1:179;                       % 180 views
p = round(sqrt(2)*N);                  % Number of rays per view
d = p - 1;
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);

%% Simulate measurement data with noise
b = A_sys * x_true;                    % Simulated sinogram (10,260 × 1)
b_noise = b+0.01*max(b)*randn(size(b));         % Add noise

%% Solve the normal equations using backslash
alpha = 600;
AtA = A_sys' * A_sys;                           % Without regularization 1,600 × 1,600
AtA_reg = AtA + alpha * speye(size(AtA));       % With regularization (alpha) 1,600 × 1,600
Atb = A_sys' * b_noise;                           % 1,600 × 1
x_rec_direct = AtA_reg \ Atb; 
figure; imagesc(reshape(x_rec_direct, N, N)); colormap gray; axis image; title('Direct Solution');% Reconstructed vector (1,600 × 1)

%% Visualize SVD to estimate alpha
[U,s,v] = svd(full(A_sys),'econ');  % Set up matrix for SVD
s=diag(s);

figure;
subplot(1,3,1);
semilogy(s) % Visualize 
subplot(1,3,2);
semilogy(abs(U'*b)) % Visualize b
subplot(1,3,3)
semilogy(abs(U'*b_noise)) % Visualize b with noise
%% Graph error to pick best alpha
alphas = 0:50:500;
relative_error = zeros(size(alphas));

for i = 1:length(alphas)
    AtA_reg = AtA + alphas(i) * speye(size(A_sys,2));
    x_rec = AtA_reg \ Atb;
    relative_error(i) = norm(x_rec - x_true)/norm(x_true); % Relative error, euclidean norm
end

figure;
semilogx(alphas, relative_error, '-o');
xlabel('\alpha'); ylabel('Error');
title('Error vs Alpha');

%% Visualize the sinogram
sinogram = reshape(b,p,length(theta)); % 57 × 180
figure;
subplot(1,2,1);
imagesc(sinogram); colormap(gray); axis xy;
xlabel('Angle index'); ylabel('Ray index');
title(sprintf('Simulated Sinogram from CT Image (alpha = %d)', alpha));

% Visualize the reconstructed image
P = reshape(x_rec, N, N);              % 40 × 40
subplot(1,2,2)
imagesc(P); colormap gray; axis image;
title(sprintf('Reconstructed Image, AIR Tools (alpha = %d)', alpha));
