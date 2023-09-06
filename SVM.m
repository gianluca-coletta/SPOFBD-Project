close all;
clear all;
clc;

% Read the clean dataset
opts = detectImportOptions('Dataset/dataset_prova_clean.csv'); 
opts.VariableNamingRule = 'preserve';  
df = readtable('Dataset/dataset_prova_clean.csv', opts);

% Dataset size
size(df)
[n,m] = size(df);

% Show first 5 rows of dataset
head(df, 5)

% Randomization of dataset rows
% df= df(randperm(size(df, 1)), :);
% head(df, 5)

% Split the dataset in training and test
labels = df(:,1);
y = table2array(labels);
x1 = df(:,2:m);
x = table2array(x1);

y= y';
x= x';

m=size(x,2);
n=size(x,1);

% Percentage samples for the training set
split = 0.80;

x_train = x(:,1:floor(m*split));
x_test = x(:,floor(m*split)+1:m);

y_train = y(1:floor(m*split));
y_test = y(floor(m*split)+1:m);

% Create matrix A
A = [ ((ones(n,1)*y_train).*x_train)' y_train'];

% SVM centralized approach
lambda = 1e-2; 

m_train = floor(m*split);
m_test = m - m_train;

cvx_begin quiet
variables x_v(n+1)
    minimize (sum( max(0, 1 - A*x_v)) + lambda*sum_square(x_v)) 
cvx_end

w_c = x_v(1:n,1);
b_c = x_v(n+1,1);

result = sign(w_c'*x_test+b_c);
Accuracy = length(find(y_test==result))/m_test

% Centralized approach evaluation

% Reshape
result_0 = result;
result_0(find(result_0==-1))=0;

y_test_0 = y_test;
y_test_0(find(y_test_0==-1))=0;

% SVM Confusion Matrix and centralized ROC 
figure
    title('Confusion Matrix SVM centralized')
    plotconfusion(y_test_0,result_0);
    shg

% ROC

% Calculate class probabilities
scores = w_c' * x_test + b_c;

% Calculate the ROC curve using perfcurve
[fpr, tpr, thresholds] = perfcurve(y_test_0, scores, 1);

% Plot the ROC curve
figure;
plot(fpr, tpr, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve SVM Centralized');

% Calculate the area under the ROC curve (AUC)
auc = trapz(fpr, tpr);
fprintf('Area under the ROC curve (AUC): %.4f\n', auc);


% Splitted by examples approach

% Reshape
result(find(result==0))=-1;
y_test(find(y_test==0))=-1;

score = []

N = 6;
iterations = 50;
p = randi([1 N], m_train,1);
for i = 1:N
    tmp{i} = A(p==i,:);
end
A_split = tmp;
X = zeros(n+1,N);
z = zeros(n+1,1);
u = zeros(n+1,N);
rho = 1e0;

score = [];
D = [];

h = waitbar(0, 'In progress...');  % Progress bar
for k=1:iterations
    for i = 1:N
        cvx_begin quiet
            variable x_v(n+1)
            minimize (sum(max(0, A_split{i}*x_v+1)) + (rho/2)*sum_square(x_v - z + u(:,i)))
        cvx_end
        X(:,i) = x_v;
    end
    
    z = (N*rho/(2*lambda + N*rho))*mean(X + u, 2);
    
    for i = 1:N
        u(:,i)=u(:,i) + X(:,i) - z;
    end
    
    % Disagreement
    P=[];
    for i = 1:N
        p = X(:,i) - mean(X,2);
        P = [P p];
    end
    
    dk = sum(sum_square(P));
    D = [D dk];
    
    x_split = X(:,1);
    w_split = x_split(1:n,1);
    b_split = x_split(n+1,1);
    
    result_split_i = -sign(w_split'*x_test+b_split);
    accuracy_split = length(find(y_test==result_split_i))/m_test;
    score = [score accuracy_split];
  
    waitbar(k / iterations, h, sprintf('Iteration %d of %d', k, iterations));
end

close(h);

result_split = -sign(w_split'*x_test+b_split);
Accuracy_split = length(find(y_test==result_split))/m_test

% Splitted approach evaluation

% Reshape
y_test_0 = y_test;
y_test_0(find(y_test_0==-1))=0;


result_split_0 = result_split;
result_split_0(find(result_split_0==-1))=0;

% Confusion matrix
figure
    title('Confusion Matrix SVM Split')
    plotconfusion(y_test_0,result_split_0);
    shg

% ROC

% Calculates class scores using the weights and bias of the splitted approach
scores_split = w_split' * x_test + b_split;

% Calculate ROC
[fpr_split, tpr_split, thresholds_split] = roc(y_test_0, scores_split);

% ROC plot
figure;
plot(fpr_split, tpr_split, 'g-', 'LineWidth', 2);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve SVM Splitted');

% AUC
auc_split = trapz(fpr_split, tpr_split);
fprintf('Area under the ROC curve (AUC) for Splitted approach: %.4f\n', auc_split);

% Centralized-example comparison

bound = Accuracy*ones(iterations, 1);

figure
subplot(2,1,1);
semilogy(D)
xlabel('Iterations')
ylabel('Disagreement')
ylim([10^(-2) 10^2])
title(['Iterations = ',num2str(iterations),' Processor = ',num2str(N),' rho = ',num2str(rho)])

subplot(2,1,2);
plot(bound,'g')
hold on

plot(score,'r')
xlabel('Iterations')
ylabel('Accuracy')
ylim([Accuracy*0.9  Accuracy*1.05])

