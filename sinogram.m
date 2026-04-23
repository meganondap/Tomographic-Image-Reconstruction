%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% view_sinogram.m
%
% View the sinogram (projection data) of a CT image.
% - Loads an image X_true
% - Builds parallel-beam system matrix A_sys using AIRToolsII
% - Computes sinogram b = A_sys * x_true(:)
% - Displays the sinogram as an image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% ---------------- USER PARAMETERS ----------------
N = 64;  % image size N x N

% Path to a single CT image (PNG, JPG, etc.)
imagePath = fullfile('C:\Users\megan\Box\MeganOndap\MATLAB Code\Images06_png\000308_03_01\053.png');  % change to your file

% Parallel-beam geometry
theta = 0:0.5:179;        % projection angles (degrees)
p     = round(sqrt(2)*N); % number of rays per projection
d     = p - 1;

%% ---------------- LOAD AND PREPARE IMAGE ----------------
img_orig = imread(imagePath);

% Convert to grayscale if needed
if ndims(img_orig) == 3
    img_gray = rgb2gray(img_orig);
else
    img_gray = img_orig;
end

% Resize to N x N
img_gray = imresize(img_gray, [N N]);

% Convert to double in [0,1]
X_true = im2double(img_gray);

%% ---------------- BUILD SYSTEM MATRIX A_sys ----------------
% Requires AIRToolsII
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);

%% ---------------- COMPUTE SINOGRAM ----------------
x_vec = X_true(:);       % N^2 x 1
b     = A_sys * x_vec;   % sinogram as a long vector

% Reshape sinogram into [num_rays, num_angles]
num_angles = numel(theta);
num_rays   = p;
B = reshape(b, [num_rays, num_angles]);

%% Plot Together

figure;
subplot(1,2,1)
imagesc(X_true); axis image off; colormap gray;
title('X\_true (input image)');

subplot(1,2,2)
imagesc(theta, 1:num_rays, B);
axis xy;
xlabel('\theta (degrees)');
ylabel('Ray index');
title('Sinogram (A\_sys * x\_true)');
colormap gray;

%% Plot system Matrix A
figure;
imagesc(A_sys);
colormap gray;
title('A sys');

