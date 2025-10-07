%% Homework 1: Question 3: Human Activity Recognition
clear all, close all,

% For Human Activity Recognition ---
pathName = "C:\Users\Action Lab\OneDrive - Northeastern University\EECE5644 Machine Learning\UCI HAR Dataset\"
% extract class categories 
activities = fullfile('C:\Users\Action Lab\OneDrive - Northeastern University\EECE5644 Machine Learning\UCI HAR Dataset\', 'activity_labels.txt');
fid = fopen(activities,'r');
C = textscan(fid, '%d %s', 'Delimiter', ' ');
fclose(fid);
classes = C{2};

% Extract features
features = fullfile('C:\Users\Action Lab\OneDrive - Northeastern University\EECE5644 Machine Learning\UCI HAR Dataset\', 'features.txt');
fid = fopen(features,'r');
C = textscan(fid, '%d %s', 'Delimiter', ' ');
fclose(fid);
feature_names = C{2};

%% 3) Load train/test matrices
fprintf('Loading train/test data (this can take a moment)...\n');
X_train = readmatrix(fullfile(pathName,'train','X_train.txt'));
y_train = readmatrix(fullfile(pathName,'train','y_train.txt'));
subj_train = readmatrix(fullfile(pathName,'train','subject_train.txt'));

X_test  = readmatrix(fullfile(pathName,'test','X_test.txt'));
y_test  = readmatrix(fullfile(pathName,'test','y_test.txt'));
subj_test = readmatrix(fullfile(pathName,'test','subject_test.txt'));

% ensure numeric
X_train = double(X_train); X_test = double(X_test);
y_train = double(y_train); y_test = double(y_test);

class_labels = unique(y_train);
C= numel(class_labels); % there are in reality only 7 classes from 3-9
[N, n] = size(X_train); % should be 7352 datapoints, 561 dimensional

% initialize mean, covariance, and priors 
mu = cell(C,1);
Sigma = cell(C,1);
priors = zeros(C,1);

% compute sample average and sample counts to get estimated mean vector and priors 
for k = 1:C
    cls = class_labels(k); % select class
    idx = (y_train == cls); % find index corresponding to class
    Xk = X_train(idx, :);  % look up sample value at those indices
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
    pxgivenl(l,:) = evalGaussianPDF(X_train', mu(:,l), Sigma_reg(:,:,l)); % Evaluate p(x|L=l)
    pxgivenl_test(l,:) = evalGaussianPDF(X_test', mu(:,l), Sigma_reg(:,:,l)); % Evaluate p(x|L=l)
end

% choose class with max logpdf
[~, kidx] = max(pxgivenl, [], 1);
ypred_train = class_labels(kidx);

% choose class with max logpdf
[~, kidx] = max(pxgivenl_test, [], 1);
ypred_test = class_labels(kidx);

% Evaluate: errors & confusion matrices
train_err = mean(ypred_train ~= y_train);
test_err  = mean(ypred_test  ~= y_test);
fprintf('\nResults:\n');
fprintf('  Training error   = %.4f  (%.2f%%)\n', train_err, 100*train_err);
fprintf('  Test error       = %.4f  (%.2f%%)\n\n', test_err, 100*test_err);


% confusion matrices
C_train = confusionmat(y_train, ypred_train, 'Order', class_labels);
Ctrain_normalized = C_train ./ sum(C_train, 2);
C_test  = confusionmat(y_test,  ypred_test,  'Order', class_labels);
Ctest_normalized = C_test ./ sum(C_test, 2);

fprintf('Confusion matrix (rows=true class, cols=predicted) — TRAIN\n');
disp(array2table(Ctrain_normalized, 'VariableNames', cellstr(classes), 'RowNames', cellstr(classes)));

fprintf('Confusion matrix (rows=true class, cols=predicted) — TEST\n');
disp(array2table(Ctest_normalized, 'VariableNames', cellstr(classes), 'RowNames', cellstr(classes)));

% PCA visualization (2D + 3D)
[coeff, score, latent, ~, explained] = pca(X_train);

figure('Name','PCA 2D (train)', 'NumberTitle','off');
gscatter(score(:,1), score(:,2), y_train);
xlabel('PC1'); ylabel('PC2');
title(sprintf('PCA 2D (train) — PC1 %.1f%%, PC2 %.1f%% explained', explained(1), explained(2)));
legend(classes,'Location','bestoutside');

figure('Name','PCA 3D (train)', 'NumberTitle','off');
scatter3(score(:,1), score(:,2), score(:,3), 12, y_train, 'filled');
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title('PCA 3D (train)');
colormap('jet'); colorbar('Ticks',1:C,'TickLabels',classes);
