function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.



%This tutorial gives a method for findClosestCentroids() by iterating through the centroids. This runs considerably faster than looping through the training examples.

%Create a "distance" matrix of size (m x K) and initialize it to all zeros. 'm' is the number of training examples, K is the number of centroids.
%Use a for-loop over the 1:K centroids.
%Inside this loop, create a column vector of the distance from each training example to that centroid, and store it as a column of the distance matrix. One method is to use the sum() function and the bsxfun() function to calculate the sum of the squares of the differences between each row in the X matrix and a centroid.
%When the for-loop ends, you'll have a matrix of centroid distances.
%Then return idx as the vector of the indexes of the locations with the minimum distance. The result is a vector of size (m x 1) with the indexes of the closest centroids.
%

% Set number of example m
m = size(X, 1);
% Initiate distance matrix
distance = zeros(m, K);

for i = 1:K
	diffs = bsxfun(@minus, X, centroids(i,:));
	distance(:,i) = sum(diffs.^2, 2);
end

[small,idx] = min(distance,[],2);
idx=idx(:);

% =============================================================

end

