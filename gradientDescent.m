function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); # (num_iters x 1) matrix


%++++++++ Vectorize Implementation of Gradient and  Theta. 
%Note Theta will change each time we we iterate 
% This gradientDescent function find the optimized Theta +++++++++
for iter = 1:num_iters
     
    %++++++++++Vectorize Implementation ++++++++++++++
    
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    delta= zeros(length(theta),1); % (n+1) x 1 matrix. where n =1   
    for i= 1: length(theta),
      delta(i) = (X*theta - y)' * X(:,i);
      
    endfor
    theta = theta - (alpha / m) * delta;   # (n+1) x 1 matrix. where n =1 






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);  

end

end
