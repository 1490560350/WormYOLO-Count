function average_length = Lengthcalculation(data)
    % 检查数据是否为空
    if isempty(data)
        error('无法读取数据，请检查数据内容。');
    end
    
    numRows = size(data, 1);  % 获取数据行数
    totalLength = 0;

    % 遍历每一行
    for li = 1:numRows
        row = data(li, :);
        % 提取点坐标
        num_points = length(row) / 2;
        points = reshape(row, 2, num_points)';
        
        % 计算每行的长度
        rowLength = 0;
        for lj = 1:num_points - 1
            % 计算相邻点之间的距离
            length_dist = sqrt((points(lj+1, 1) - points(lj, 1))^2 + (points(lj+1, 2) - points(lj, 2))^2);
            rowLength = rowLength + length_dist;
        end

        % 累加所有行的长度
        totalLength = totalLength + rowLength;
    end

    % 计算平均长度
    average_length = totalLength / numRows;
end
