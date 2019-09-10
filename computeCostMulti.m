function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

%++++++++++Vectorize Implementation ++++++++++++++

% Initialize some useful values
m = length(y); % number of training examples

% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

error= (X * theta - y) .^2; #X is m x (n+1) matrix. Here ".^2" is element wise square.
J = (1/(2 * m)) * sum(error) ;

end
