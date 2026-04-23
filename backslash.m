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


%% ---------------- BUILD A_sys AND SIMULATE DATA ----------------
[A_sys, ~, ~, theta, p, d] = paralleltomo(N, theta, p, d);

x_true_vec = X_true(:);
b = A_sys * x_true_vec;  % simulated sinogram

%% ---------------- DIRECT RECONSTRUCTION ----------------
% Solve least-squares problem using backslash
x_rec_vec = A_sys \ b;
X_rec = reshape(x_rec_vec, [N N]);

X_diff = X_true-X_rec;

figure;
subplot(1,3,1);
imagesc(X_true); axis image off; colormap gray;
title('X\_true');


subplot(1,3,2);
imagesc(X_rec); axis image off; colormap gray;
title('Backslash reconstruction');
sgtitle('Direct reconstruction with MATLAB backslash');

subplot(1,3,3);
imagesc(X_diff); axis image off; colormap gray;
title('X\_true - X\_rec');
