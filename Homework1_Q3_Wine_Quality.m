%% Homework 1: Question 3: Wine Quality
%% Leonora Blodgett
clear all, close all,

% --- For Wine quality ---
T = readtable('winequality-white.csv');
X = T{:,1:11};  % first 11 columns are features
labels = T{:,12};   % the quality score 0-10 
classes = unique(labels);
C= numel(classes); % there are in reality only 7 classes from 3-9
N = 4898; % Number of samples
n = 11; % 11 dimensional (11 features)

% initialize mean, covariance, and priors 
mu = cell(C,1);
Sigma = cell(C,1);
priors = zeros(C,1);

for k = 1:C
    % compute sample average and sample counts to get estimated mean vector and priors 
    cls = classes(k); % select class
    idx = (labels == cls); % find index corresponding to class
    Xk = X(idx, :);  % look up sample value at those indices
    Nk = sum(idx); % find the total number of instances in that class
    priors(k) = Nk / N; % priors = class instances / total number of data samples
    mu{k} = mean(Xk, 1)';  % mean vector is sample average of data for each class

    % estimate sample covariance: (1/Nk) Σ (x - mu)(x - mu)'
    Xk0 = Xk - repmat(mu{k}', Nk, 1);
    C_sampavg = (Xk0' * Xk0) / Nk;

    % Regularization step 
    alpha = 0.01;  % 0 < alpha < 1
    lambda = alpha * trace(C_sampavg) / rank(C_sampavg);
    % estimate sample covariance: C_regularized = C_sampavg + lambda*I
    Sigma_reg{k} = C_sampavg + lambda * eye(n);
end

% Step 1: Convert the 7x1 cell array into a single column vector of size 77x1
combined = cell2mat(mu);  % Now it's 77x1 (since 7*11 = 77)
% Step 2: Reshape it to 11x7
mu = reshape(combined, n, C);
Sigma_reg = cat(3, Sigma_reg{:});
priors = priors';
    
logpdf = zeros(N, C);

% Shared computation for both parts
for l = 1:C
    pxgivenl(l,:) = evalGaussianPDF(X', mu(:,l), Sigma_reg(:,:,l)); % Evaluate p(x|L=l)
end


% choose class with max logpdf
[~, kidx] = max(pxgivenl, [], 1);
ypred = classes(kidx);

M = numel(labels);
Cmat = confusionmat(labels, ypred, 'Order', classes);
nerr = sum(labels ~= ypred);
Pe = nerr / M;
cm_normalized = Cmat ./ sum(Cmat, 2);
disp('Confusion matrix normalized:');

labels = labels';
figure(1)
subplot(4,1,1)
gscatter(X(:,1), X(:,2), labels)
xlabel("fixed acidity")
ylabel("volatile acidity")
subplot(4,1,2)
gscatter(X(:,3), X(:,4), labels)
xlabel("citric acid")
ylabel("residual sugar")
subplot(4,1,3)
gscatter(X(:,5), X(:,6), labels)
xlabel("chlorides")
ylabel("free sulfur dioxide")
subplot(4,1,4)
gscatter(X(:,7), X(:,8), labels)
xlabel("Total sulfur dioxide")
ylabel("Density")

[coeff, score, ~, ~, explained] = pca(X);  % use built in matlab function
% score is N×d, first few columns are principal components
figure(2); gscatter(score(:,1), score(:,2), labels);
xlabel('PC1'); ylabel('PC2');
xlim([-170, 250])
ylim([-75, 100])
title('Projection onto first 2 PCs, colored by true class');
