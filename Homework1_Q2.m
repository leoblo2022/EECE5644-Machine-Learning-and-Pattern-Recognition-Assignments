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
% Define specific 2x2 covariance matrices (as a good example for the write-up)
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
mShapes = {'o', '^', '+', 'd'}; % Circle, square, triangle, diamond
color_green = {[0 1 0], [0 0.6 0.4], [0.4 0.6 0], [0.2 0.6 0.2]};
colors = zeros(N, 3); % RGB colors
correct = (decisions == true_labels);
figure(2)
hold on

for i = 1:N
    if correct(i)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Question 2, Part B: Using a different loss matrix 

px = gmmParameters.priors*pxgivenl; % Total probability theorem
classPosteriors = pxgivenl.*repmat(gmmParameters.priors',1,N)./repmat(px,4,1); % P(L=l|x)
C=4; % number of classes
lossMatrix = [0 10 10 100; 1 0 10 100; 1 1 0 100; 1 1 1 0]; % Define the loss matrix (NOT 0-1 loss)
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)

% ERM decision (minimum expected risk)
[~,erm_decisions] = min(expectedRisks,[],1); % Minimum expected risk decision with 0-1 loss is 

% Step 6: Compute average risk using sample average 
average_risk = 0;
for n = 1:N
    i = erm_decisions(n);     % Decision
    j = true_labels(n);       % True label
    average_risk = average_risk + lossMatrix(i,j);
end

average_risk = average_risk /N;
fprintf('Average Expected Risk under ERM: %.4f\n', average_risk);

for d = 1:C % each decision option
    for l = 1:C % each class label
        ind_dl = find(erm_decisions==d & labels==l);
        ConfusionMatrix_partB(d,l) = length(ind_dl)/length(find(labels==l));
    end
end
disp('Confusion matrix with our defined loss matrix (not 0-1 loss):');
disp(ConfusionMatrix_partB)


% === ERM Classification Scatter Plot ===
figure(3);
hold on

% Loop over samples and plot each with marker and color
for i = 1:N
    marker = mShapes{true_labels(i)};
    is_correct = (erm_decisions(i) == true_labels(i));
    if is_correct
        color = color_green{true_labels(i)};
    else
        color = [0.8, 0, 0]; % Red for incorrect
    end
    
    % Plot sample
    plot(X(1,i), X(2,i), marker, 'MarkerEdgeColor', color, ...
        'MarkerFaceColor', color, 'MarkerSize', 4);
end

xlabel('x_1');
ylabel('x_2');
title('ERM Classification Results (Part B)');
legend('Class 1', 'Class 2', 'Class 3', 'Class 4');
hold off;


