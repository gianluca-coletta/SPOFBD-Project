close all;
clear all;
clc;

% Read the clean dataset
opts = detectImportOptions('dataset_clean.csv'); 
opts.VariableNamingRule = 'preserve';  
df = readtable('dataset_clean.csv', opts);

% Show first 5 rows of dataset
head(df, 5)

% Randomization of dataset rows
df= df(randperm(size(df, 1)), :);
head(df, 5)

% Split the dataset in training and test
% labels = df(:,1);
% y = table2array(labels);
% x1 = df(:,2:m);
% x = table2array(x1);
