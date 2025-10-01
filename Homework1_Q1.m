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
%% Question 1, Part B: 
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
