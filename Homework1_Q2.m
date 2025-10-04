%% Homework 1: Question 2
%% Leonora Blodgett
clear all, close all,

%% Question 2, Part A: Minimum probability of error classification (0-1 loss, MAP classification rule)
C=4;
N = 10000; % Number of samples
n = 2; % number of feature dimensions
gmmParameters.priors = ones(1,C)/C; % uniform priors, in this case 0.25
gmmParameters.meanVectors = 3*1.25*n*C*(rand(n,C)); % arbitrary mean vectors
for l = 1:C
    A = 5*eye(n)+0.2*randn(n,n);
    gmmParameters.covMatrices(:,:,l) = A'*A; % arbitrary covariance matrices
end

%{
% Define specific 2x1 means vectors (as a good example for the write-up)
mu1 = [0; 0]; mu2 = [2; 5]; mu3 = [-1; 3.2]; mu4 = [3; -1.5];
mu(:,1) = mu1; mu(:,2) = mu2; mu(:,3) = mu3; mu(:,4) = mu4;  
% Define arbitrary 2x2 covariance matrices (as a good example for the write-up)
Sigma1 = [1 0.2; 0.2 1]; Sigma2 = [1 -0.4; -0.4 1]; Sigma3 = [0.3 0; 0 0.3]; Sigma4 = [1 0.8; 0.8 1];
Sigma(:,:,1) = Sigma1; Sigma(:,:,2) = Sigma2; Sigma(:,:,3) = Sigma3; Sigma(:,:,4) = Sigma4;
for l = 1:C
    gmmParameters.meanVectors(:,l) = mu(:, l);
    gmmParameters.covMatrices(:,:,l) = Sigma(:,:,l); % arbitrary covariance matrices
end
%}

% Generate data from specified pdf
[X,labels] = generateDataFromGMM(N,gmmParameters); % Generate data
for l = 1:C
    Nclass(l,1) = length(find(labels==l));
end


% Shared computation for both parts
for l = 1:C
    pxgivenl(l,:) = evalGaussianPDF(X,gmmParameters.meanVectors(:,l),gmmParameters.covMatrices(:,:,l)); % Evaluate p(x|L=l)
end
true_labels = labels;


px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,4,1); % P(L=l|x)
C=4; % number of classes
lossMatrix = ones(C,C)-eye(C); % For min-Perror design, use 0-1 loss
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)

% MAP decision
[~,decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is 

for d = 1:C % each decision option
    for l = 1:C % each class label
        ind_dl = find(decisions==d & labels==l);
        ConfusionMatrix(d,l) = length(ind_dl)/length(find(labels==l));
    end
end
disp('Confusion matrix:');
disp(ConfusionMatrix)

% Define markers for each true label
mShapes = {'o', 'x', '+', '*'}; % Circle, square, triangle, diamond
color_green = {[0 1 0], [0 0.7 0.3], [0.3 0.7 0], [0.2 0.7 0.2]};
colors = zeros(N, 3); % RGB colors
correct = (decisions == true_labels);
figure(2)
hold on

for i = 1:N
    if correct(i)
        %col = [0 1 0]; % green
        col = color_green{true_labels(i)};
    else
        col = [1 0 0]; % red
    end
    marker = mShapes{true_labels(i)};
    plot(X(1,i), X(2,i), marker, 'Color', col, 'MarkerSize', 4);
end

legend('Class 1', 'Class 2', 'Class 3', 'Class 4');
xlabel('x1');
ylabel('x2');
title('MAP Classification Results: Marker = True Label, Color = Classification Accuracy');
hold off;

