
%% Initialization
clear ; close all; clc

%% ==================== Part 1: Email Preprocessing ====================
%  To use Naive Bayes to classify emails into Spam v.s. Non-Spam.
% Extract Features

numTrainDocs = 700;
numTokens = 2500;
M = dlmread('train-features.txt', ' ');
spmatrix = sparse(M(:,1), M(:,2), M(:,3), numTrainDocs, numTokens);
train_matrix = full(spmatrix);
train_labels = dlmread('train-labels.txt');

% Spam and ham indexes

num_spam = find(train_labels == 1);
num_ham = find(train_labels == 0);
% Probability of an email being spam or ham

py1 = sum(train_labels(num_spam))/numTrainDocs;
py0 = sum(train_labels(num_ham)+1)/numTrainDocs;

% 
email_lengths = sum(train_matrix, 2);
Numberofspam = sum(email_lengths(num_spam));

Numberofham = sum(email_lengths(num_ham));

% Probability P(x|y = 1) and P(x|y = 0) with Laplas smoothing

Px_y1 = (sum(train_matrix(num_spam, :)) + 1) ./(Numberofspam + numTokens);

Px_y0 = (sum(train_matrix(num_ham, :)) + 1) ./(Numberofham + numTokens);

% read the test matrix in the same way we read the training matrix
N = dlmread('test-features.txt', ' ');
spmatrix = sparse(N(:,1), N(:,2), N(:,3));
test_matrix = full(spmatrix);

% Store the number of test documents and the size of the dictionary
numTestDocs = size(test_matrix, 1);
numTokens = size(test_matrix, 2);

% The output vector is a vector that will store the spam/nonspam prediction
% for the documents in our test set.
output = zeros(numTestDocs, 1);

% Calculate log p(x|y=1) + log p(y=1)
% and log p(x|y=0) + log p(y=0)
% for every document
% make your prediction based on what value is higher


log_a = test_matrix*(log(Px_y1))' + log(py1);
log_b = test_matrix*(log(Px_y0))'+ log(1 - py1);  
output = log_a > log_b;

% Read the correct labels of the test set
test_labels = dlmread('test-labels.txt');

% Compute the error on the test set
% A document is misclassified if it's predicted label is different from
% the actual label, so count the number of 1's from an exclusive "or"

numdocs_wrong = sum(xor(output, test_labels))

%Print out error statistics on the test set
fraction_wrong = numdocs_wrong/numTestDocs
