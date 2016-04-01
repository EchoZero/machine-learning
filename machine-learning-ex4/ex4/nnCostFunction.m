function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%% Expand y output values into a matrix of single values
y_matrix = eye(num_labels)(y, :);
%% Perfrom the forward propagation
%% a1 =  equals the X input matrix with a column of 1's added (bias units) as the first column.
a1 = [ones(m,1), X];
%% z2 equals the product of a1 and Θ1
z2 = a1 * Theta1';
%% a2 is the result of passing z2 through g()
a2 = sigmoid(z2);
%% Then add a column of bias units to a2 (as the first column)
a2 = [ones(size(a2,1),1) a2];
z3 = a2 * Theta2';
%% a3 is the result of passing z3 through g()
a3 = sigmoid(z3);

%% Calculate unregularized J, y_matrix(5000*10), h=a3 (5000*3), so use element-wise to get scalar value
%% along with the double sum
fp = ( -y_matrix) .* log(a3);
sp = (1 - y_matrix) .* log(1 - a3);

J = sum(sum(1/m * (fp - sp))); 


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%% d3 is the difference between a3 and the y_matrix. 
%% The dimensions are the same as both, (m x r).
d3 = a3 - y_matrix;
%% d2 is tricky. It uses the (:,2:end) columns of Theta2. 
%% d2 is the product of d3 and Theta2(no bias),
%% then element-wise scaled by sigmoid gradient of z2.
%% The size is (m x r) ⋅ (r x h) --> (m x h). The size is the same as z2, as must be.
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2);

%% Delta1 is the product of d2 and a1
Delta1 = d2' * a1;
Delta2 = d3' * a2;

%%Then gradient si Delta divided by m

Theta1_grad = Delta1./m;
Theta2_grad = Delta2./m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Regularization for cost function
%% First exlucude bias unit from Theta1 Theta2

T1 = Theta1(:, 2:end);
T2 = Theta2(:, 2:end);
%% Then Calculate the double sum of square
square1 = sum(sum(T1.^2));
square2 = sum(sum(T2.^2));
%% Add the regularized part to the origin
regu_part = (lambda/ (2*m)) * (square1 + square2);
J =  J + regu_part;

%% Regularization for gradient part
Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = Theta1_grad + ((lambda/m) .* Theta1);
Theta2_grad = Theta2_grad + ((lambda/m) .* Theta2);

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
