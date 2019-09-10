function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

%++++++++++Vectorize Implementation ++++++++++++++
theta = pinv(X' * X) *X' *y;  # (n+1) x 1 matrix
end
