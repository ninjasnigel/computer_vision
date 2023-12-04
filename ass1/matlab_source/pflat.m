function result = pflat(x)
    % pflat - Projective flattening of homogeneous coordinates
    %
    % This function divides all but the last row of the input matrix 'x'
    % by its last row, effectively performing a projective division.
    %
    % Syntax:  result = pflat(x)
    %
    % Inputs:
    %    x - A matrix of size (n x m), where n > 1, representing homogeneous coordinates.
    %
    % Outputs:
    %    result - A matrix of size ((n-1) x m), with each element of the first (n-1) rows
    %             of 'x' divided by the corresponding element of the last row of 'x'.

    % Safety check for the input dimensions
    if size(x,1) <= 1
        error('Input matrix must have more than one row.');
    end

    % Perform the division
    result = x(1:end-1,:) ./ x(end,:);
end