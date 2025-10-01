%% Homework 1: Question 1
%% Leonora Blodgett
clear all, close all,

N = 10000; % Number of samples
p0 = 0.65; p1 = 0.35; %class prior probabilities
u = rand(1,N)>=p0; N0 = length(find(u==0)); N1 = length(find(u==1));
mu0 = [-1/2;-1/2;-1/2]; Sigma0 = [1,-0.5,0.3;-0.5,1,-0.5;0.3,-0.5,1];
r0 = mvnrnd(mu0, Sigma0, N0);
figure(1), plot3(r0(:,1),r0(:,2),r0(:,3),'.b'); axis equal, hold on,
mu1 = [1;1;1]; Sigma1 = [1,0.3,-0.2;0.3,1,0.3;-0.2,0.3,1];
r1 = mvnrnd(mu1, Sigma1, N1);
figure(1), plot3(r1(:,1),r1(:,2),r1(:,3),'.r'); axis equal, hold on,

%% Question 1, Part A, section 2: Plot ROC curve 
n = 3; % number of feature dimensions
mu(:,1) = [-1/2;-1/2;-1/2]; mu(:,2) = [1;1;1];
Sigma(:,:,1) = [1,-0.5,0.3;-0.5,1,-0.5;0.3,-0.5,1]; Sigma(:,:,2) = [1,0.3,-0.2;0.3,1,0.3;-0.2,0.3,1];
p = [p0, p1];
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
X = zeros(n,N); % save up space

for l = 0:1
    X(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

% Evaluate log-likelihoods
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(X,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(X,mu(:,1),Sigma(:,:,1)));
decision = (discriminantScore >= log(gamma)); % LDA threshold not optimized to minimize its own E[Risk]!

% For ROC curve, counts FP, FN, TN, TP and errors 
[sortedScores,ind] = sort(discriminantScore,'ascend');
% Sweep gamma (tau) values from 0 to infinity
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScore >= tau);
    Ptn(i) = length(find(decisions==0 & label==0))/length(find(label==0)); % True Negative (decide 0 when L=0)
    Ptp(i) = length(find(decisions==1 & label==1))/length(find(label==1)); % True Positive (decide 1 when L=1)
    Pfp(i) = length(find(decisions==1 & label==0))/length(find(label==0)); % False Positive (decide 1 when L=0)
    Pfn(i) = length(find(decisions==0 & label==1))/length(find(label==1)); % False Negative (decide 0 when L=1)
    Perror(i) = sum(decisions~=label)/length(label); % Errors (decide 1 when L=0 or decide 0 when L=1)
end

% Plot ROC curve 
figure(3)
plot(Pfp,Ptp,'b.-','LineWidth',1.5),
xlabel('P(False+)'),ylabel('P(True+)'), title('ROC Curve for ERM Discriminant Scores'),

%% Question 1, Part A, section 3: 
% Find minimum error and threshold
[Pe_min, idxMin] = min(Perror);
optimized_tau = thresholdList(idxMin);
Ptp_emp = Ptp(idxMin);
Pfp_emp = Pfp(idxMin);

% Plot ROC again with the minimum-Perror point marked
figure(4);
plot(Pfp,Ptp,'b.-','LineWidth',1.5); hold on;
plot(Pfp_emp, Ptp_emp, 'go','MarkerSize',10,'LineWidth',2); % mark empirical min error
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve with Minimum-Error Operating Point');
axis([0 1 0 1]);


% Plot Perror with the minimum-Perror point marked
figure(5)
plot(thresholdList,Perror,'*m'); hold on;
plot(optimized_tau, Perror(idxMin), 'go','MarkerSize',10,'LineWidth',2); % mark empirical min error
xlabel('Thresholds'), ylabel('P(error) for ERM Discriminant Scores'),

% Print results
fprintf('Empirical min P(error) = %.4f at gamma = %.4f\n', Pe_min, exp(optimized_tau));
fprintf('Theoretical optimal gamma = %.4f\n', p0/p1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Question 1, Part B: Naive Bayesian Classifier
 % Falsely assume covariance matrix is identity matrix
Sigma(:,:,1) = eye(3); Sigma(:,:,2) = eye(3);

% repeat all the same steps as part A
for l = 0:1
    X(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

% Evaluate log-likelihoods
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(X,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(X,mu(:,1),Sigma(:,:,1)));
decision = (discriminantScore >= log(gamma)); % LDA threshold not optimized to minimize its own E[Risk]!

% For ROC curve, counts FP, FN, TN, TP and errors 
[sortedScores,ind] = sort(discriminantScore,'ascend');
% Sweep gamma (tau) values from 0 to infinity
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScore >= tau);
    Ptn(i) = length(find(decisions==0 & label==0))/length(find(label==0)); % True Negative (decide 0 when L=0)
    Ptp(i) = length(find(decisions==1 & label==1))/length(find(label==1)); % True Positive (decide 1 when L=1)
    Pfp(i) = length(find(decisions==1 & label==0))/length(find(label==0)); % False Positive (decide 1 when L=0)
    Pfn(i) = length(find(decisions==0 & label==1))/length(find(label==1)); % False Negative (decide 0 when L=1)
    Perror(i) = sum(decisions~=label)/length(label); % Errors (decide 1 when L=0 or decide 0 when L=1)
end


% Find minimum error and threshold
[Pe_min, idxMin] = min(Perror);
optimized_tau = thresholdList(idxMin);
Ptp_emp = Ptp(idxMin);
Pfp_emp = Pfp(idxMin);

% Plot ROC with the minimum-Perror point marked
figure(4);
plot(Pfp,Ptp,'c.-','LineWidth',1.5); hold on;
plot(Pfp_emp, Ptp_emp, 'go','MarkerSize',10,'LineWidth',2); % mark empirical min error
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve with Minimum-Error Operating Point');
axis([0 1 0 1]);
legend("Correct ERM", "Minimum Perror ERM", "Naive Bayesian Classifier", "Minimum Perror NB")


% Plot Perror with the minimum-Perror point marked
figure(5)
plot(thresholdList,Perror,'*r'); hold on;
plot(optimized_tau, Perror(idxMin), 'go','MarkerSize',10,'LineWidth',2); % mark empirical min error
xlabel('Thresholds'), ylabel('P(error) for ERM Discriminant Scores'),
legend("Correct ERM", "Minimum Perror ERM", "Naive Bayesian Classifier", "Minimum Perror NB")

% Print results
fprintf('Empirical min P(error) = %.4f at gamma = %.4f\n', Pe_min, exp(optimized_tau));
fprintf('Theoretical optimal gamma = %.4f\n', p0/p1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Question 1, Part C: Fisher LDA Classifier 
%% 1) Estimate class means and covariances from the 10k samples (sample averages)
label = label';
X=X';
idx0 = (label==0);
idx1 = (label==1);

x1 = X(idx0,:);
x2 = X(idx1,:);

n0 = size(x1,1);
n1 = size(x2,1);

% Estimate mean vectors and covariance matrices from samples
mu1hat = mean(x1,1)';  S1hat = cov(x1);  
mu2hat = mean(x2,1)';  S2hat = cov(x2);

% Calculate the between/within-class scatter matrices:
Sw = S2hat + S1hat;   % within-class scatter (equal weight)
Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)'; % Between-class scatter 

% Solve for the Fisher LDA projection vector (in w)
% Solve Sb * v = lambda * Sw * v  -> pick eigenvector with largest eigenvalue
[V,D] = eig(inv(Sw)*Sb); % generalized eigenproblem
% eig may produce complex small nums due to numeric; keep real parts
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

% Normalize w for convenience (direction only matters)
w = w / norm(w);

% For a two-class case, direction is proportional to inv(Sw)*(mu1-mu0)
% (sanity check) w_alt = Sw \ diffm; w_alt = w_alt / norm(w_alt);

%% 3) Project data onto w and sweep thresholds tau
proj = X * w;             % N x 1 scalar projections
proj0 = proj(idx0);
proj1 = proj(idx1);

% Choose threshold grid that covers range of projections
tauVals = linspace(min(proj)-1e-3, max(proj)+1e-3, 1000);  % dense sweep
TPR_lda = zeros(size(tauVals));
FPR_lda = zeros(size(tauVals));
Pe_lda  = zeros(size(tauVals));

for i = 1:length(tauVals)
    tau = tauVals(i);
    decisions = proj > tau;  % decide class 1 if w'*x > tau
    
    TP = sum(decisions==1 & label==1);
    FN = sum(decisions==0 & label==1);
    FP = sum(decisions==1 & label==0);
    TN = sum(decisions==0 & label==0);
    
    TPR_lda(i) = TP / (TP + FN);
    FPR_lda(i) = FP / (FP + TN);
    Pe_lda(i)  = (FP + FN) / N;
end

% Find minimum empirical error and its tau
[Pe_min_lda, idxMin_lda] = min(Pe_lda);
tau_emp = tauVals(idxMin_lda);
TPR_emp_lda = TPR_lda(idxMin_lda);
FPR_emp_lda = FPR_lda(idxMin_lda);


% Plot ROC
figure(6);
plot(FPR_lda, TPR_lda, 'g.-','LineWidth',1.5); hold on;
plot(FPR_emp_lda, TPR_emp_lda, 'mo','MarkerSize',10,'LineWidth',2);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve - Fisher Linear Discriminant Analysis');
axis([0 1 0 1]);

% Plot Perror with the minimum-Perror point marked
figure(7)
plot(tauVals,Pe_lda, '*r'); hold on;
plot(tau_emp, min(Pe_lda), 'go','MarkerSize',10,'LineWidth',2); % mark empirical min error
xlabel('Thresholds'), ylabel('P(error) for Fosher Linear Discriminant Analysis'),
%legend("Correct ERM", "Minimum Perror ERM", "Naive Bayesian Classifier", "Minimum Perror NB")

