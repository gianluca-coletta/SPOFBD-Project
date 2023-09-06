close all;
clear all;
clc;

% Read the clean dataset
opts = detectImportOptions('Dataset/dataset_clean.csv'); 
opts.VariableNamingRule = 'preserve';  
df = readtable('Dataset/dataset_clean.csv', opts);

% Dataset size
size(df)
[n,m] = size(df);

% Show first 5 rows of dataset
head(df, 5)

% Randomization of dataset rows
df= df(randperm(size(df, 1)), :);
head(df, 5)

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
split = 0.8;

x_train = x(:,1:floor(m*split));
x_test = x(:,floor(m*split)+1:m);

y_train = y(1:floor(m*split));
y_test = y(floor(m*split)+1:m);

% Create matrix A
A = [ ((ones(n,1)*y_train).*x_train)' y_train'];

% SVM centralized approach
lambda = 1e-4; 

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

figure
    title('ROC SVM centralized')
    plotroc(y_test_0,result_0);
    shg


% Distributed approach

% Reshape
result(find(result==0))=-1;
y_test(find(y_test==0))=-1;

score = []

N = 6;
iterations = 100;

