function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


[J1,grad1] = costFunction(theta,X,y);

J = J1 + (lambda/(2*m))* sum(theta(2:length(theta)).*theta(2:length(theta)));
%grad1(2:length(grad1),1)


% tempX = X(2:length(X),:);
% tempY = y(2:length(y));

% % size(X)
% % size(tempX)

% % size(y)
% % size(tempY)

% % size(theta)
% % size(tempTheta)

% [J1,grad2] = costFunction(theta,tempX,tempY);


% grad(1,1) = grad1(1,1);
% grad(2:length(grad),1) = grad2(2:length(grad2)) + (lambda/m)*theta(2:length(theta));



grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );

grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';


% =============================================================

end
