function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% setup the input layer 5000 x 401
X = [ ones(m, 1) X ];

% hidden layer
% [5000 x 401] * [25 * 401] = [5000 x 25] (25 predictions for each training set)
Z1 = X * Theta1';
a2 = sigmoid(Z1);
% add a2(0) = 1
a2 = [ones(size(a2,1), 1) a2];

% output layer
% [5000 x 26] * [10 * 26] = [5000 x 10] (10 different predictions for each previous predictions)
Z2 = a2 * Theta2';
% pick the max column [5000 x 1]
[ia3, p] = max(sigmoid(Z2), [], 2);

% =========================================================================


end
