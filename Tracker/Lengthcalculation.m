function average_length = Lengthcalculation(data)

    if isempty(data)
        error('Unable to read data. Please check the data content');
    end
    
    numRows = size(data, 1);  
    totalLength = 0;


    for li = 1:numRows
        row = data(li, :);

        num_points = length(row) / 2;
        points = reshape(row, 2, num_points)';

        rowLength = 0;
        for lj = 1:num_points - 1

            length_dist = sqrt((points(lj+1, 1) - points(lj, 1))^2 + (points(lj+1, 2) - points(lj, 2))^2);
            rowLength = rowLength + length_dist;
        end

        totalLength = totalLength + rowLength;
    end

    average_length = totalLength / numRows;
end
