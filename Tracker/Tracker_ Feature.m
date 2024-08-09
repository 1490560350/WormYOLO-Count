clc;
close all;
clear all;

trainPath = 'E:\data3\tierpsy-tracker\data\AVI_VIDEOS\1\AVI_VIDEOS_4\1_2\result\';

HT = xlsread(fullfile(trainPath, 'analysis result', 'HeadTailReg.csv'));
Ph = xlsread(fullfile(trainPath, 'analysis result', 'Pharynx.csv'));
PP = xlsread(fullfile(trainPath, 'analysis result', 'PeakPoints.csv'));
IP = xlsread(fullfile(trainPath, 'analysis result', 'InflectionPoints.csv'));
SC = xlsread(fullfile(trainPath, 'analysis result', 'Center.csv'));

rowsToKeep = getValidRows(HT, Ph, PP, IP, SC);


HT = HT(rowsToKeep, :);
Ph = Ph(rowsToKeep, :);
PP = PP(rowsToKeep, :);
IP = IP(rowsToKeep, :);
SC = SC(rowsToKeep, :);

deletedRows = find(~rowsToKeep);
if ~isempty(deletedRows)
    for i = 1:length(deletedRows)
        fprintf('Missing data in line %d of a file in the input file. The data in the corresponding lines of the other files is also deleted, the remaining data is not affected.\n', deletedRows(i));
    end
end
num = size(HT, 1);

wlength = Lengthcalculation(SC);

head = HT(:,1:2);
headt = SC(:,41:42);
tail = HT(:,3:4);
tailh = SC(:,201:202);
pharynx = Ph(:,1:2);

maxDist = [];
maxHDist = [];
maxTDist = [];
bodyBendnum = 0;
headBendnum = 0;
tailBendnum = 0;
maxspeed = 0;
Omega = 0;


for k = 1:num
    [dists, hdists, tdists] = processData(PP(k, :), IP(k, :), pharynx(k, :), tail(k, :), tailh(k, :), head(k, :), headt(k, :), SC(k, :));
    
    if ~isempty(dists)
        [maxDistValue, index1] = max(abs(dists));
        maxDist(k) = dists(index1);
    end

    if ~isempty(maxDist) && maxDist(k) > wlength / 5
        Omega = Omega + 1;
    end
    
    if ~isempty(hdists)
        maxHDistValue = max(abs(hdists));
        index2 = find(abs(hdists) == maxHDistValue, 1);
        maxHDist(k) = hdists(index2);
    end
    if ~isempty(tdists)
        maxTDistValue = max(abs(tdists));
        index3 = find(abs(tdists) == maxTDistValue, 1);
        maxTDist(k) = tdists(index3);
    end
end

% Analyze and save results
[bodyBendnum, headBendnum, tailBendnum, maxspeed] = analyzeBends(maxDist, maxHDist, maxTDist,  trainPath, bodyBendnum, headBendnum, tailBendnum, wlength);
fprintf('Dist: the number of Head bends are %d\n', headBendnum);
fprintf('Dist: the number of Body bends are %d\n', bodyBendnum);
fprintf('Dist: the number of Tail bends are %d\n', tailBendnum);

saveFeatures(fullfile(trainPath, 'analysis result', 'Features.csv'), wlength, bodyBendnum, headBendnum, tailBendnum, maxspeed, Omega);

function rowsToKeep = getCommonRows(varargin)

    numFiles = length(varargin);
    rowsToKeep = true(size(varargin{1}, 1), 1);
    for i = 1:numFiles
        currentData = varargin{i};
        validRows = ~all(isnan(currentData), 2);
        rowsToKeep = rowsToKeep & validRows;
    end
end

function cleanedData = removeEmptyRows(data)
    % Removes rows from data where all elements are NaN
    cleanedData = data(~all(isnan(data), 2), :);
end

function [dists, hdists, tdists] = processData(PP_row, IP_row, pharynx_row, tail_row, tailh_row, head_row, headt_row, SC_row)
    dists = [];
    hdists = [];
    tdists = [];
    [m, n] = size(PP_row);
    for i = 1:2:n-1
        if isnan(PP_row(i))
            continue;
        end
        
        flag = bendDirection(pharynx_row(1), pharynx_row(2), tail_row(1), tail_row(2), PP_row(i), PP_row(i+1));         
        dist = Dist(PP_row(i), PP_row(i+1), pharynx_row(1), pharynx_row(2), tail_row(1), tail_row(2)); 
        dist = dist * flag;
        dists = [dists, dist];
    end
    
    for j = 3:2:39
        if isnan(SC_row(j))
            continue;
        end
        
        hflag = bendDirection(head_row(1), head_row(2), headt_row(1), headt_row(2), SC_row(j), SC_row(j+1)); 
        hdist = Dist(SC_row(j), SC_row(j+1), head_row(1), head_row(2), headt_row(1), headt_row(2)); 
        hdist = hdist * hflag;
        hdists = [hdists, hdist];
    end
    for z = 203:2:239
        if isnan(SC_row(z))
            continue;
        end
        
        tflag = bendDirection(tailh_row(1), tailh_row(2), tail_row(1), tail_row(2), SC_row(z), SC_row(z+1)); 
        tdist = Dist(SC_row(z), SC_row(z+1), tailh_row(1), tailh_row(2), tail_row(1), tail_row(2)); 
        tdist = tdist * tflag;
        tdists = [tdists, tdist];
    end
end

function [bodyBendnum, headBendnum, tailBendnum, maxspeed] = analyzeBends(maxDist, maxHDist, maxTDist,   trainPath, bodyBendnum, headBendnum, tailBendnum, wlength)
    [bodyBendnum, maxspeed] = analyzebodyBends(maxDist,   trainPath, bodyBendnum, wlength);
    [headBendnum] = analyzeHeadBends(maxHDist, trainPath, headBendnum, wlength);
    [tailBendnum] = analyzeTailBends(maxTDist, trainPath, tailBendnum, wlength);
end

function [bendnum, maxspeed] = analyzebodyBends(distArray,   trainPath, bendnum, wlength)
    t = 1;
    i = 2;
    maxspeed = 0;
    while i <= length(distArray) - 2
        if sign(distArray(t)) * sign(distArray(i)) == 1 || (sign(distArray(t)) * sign(distArray(i)) == -1 && sign(distArray(i+1)) * sign(distArray(i)) == -1)||distArray(t) * distArray(i) == 0
            i = i + 1; 
        elseif sign(distArray(t)) * sign(distArray(i)) == -1 && sign(distArray(i+1)) * sign(distArray(i)) == 1
            t = i;
            [positiveCount, negativeCount, maxPositive, maxNegative, max1, max2] = deal(0, 0, 0, 0, 0, 0);

            while i + 2 <= length(distArray)
                if distArray(i) > 0
                    positiveCount = positiveCount + 1;
                    if distArray(i) > maxPositive
                        maxPositive = distArray(i);
                        max1 = i;
                    end
                elseif distArray(i) < 0
                    negativeCount = negativeCount + 1;  
                    if abs(distArray(i)) > abs(maxNegative)
                        maxNegative = distArray(i);
                        max2 = i;
                    end
                end

                if sign(distArray(t)) * sign(distArray(i+1)) == -1
                    if (i + 1 - t > 1) && (sign(distArray(t)) * sign(distArray(i+2)) == -1)
                        break;
                    end
                end
                i = i + 1;    
            end

            if i + 1 == length(distArray)
                if distArray(i) > 0
                    positiveCount = positiveCount + 1;
                    if distArray(i) > maxPositive
                        maxPositive = distArray(i);
                        max1 = i;
                    end
                elseif distArray(i) < 0
                    negativeCount = negativeCount + 1;  
                    if abs(distArray(i)) > abs(maxNegative)
                        maxNegative = distArray(i);
                        max2 = i;
                    end
                end

                if distArray(i+1) > 0
                    positiveCount = positiveCount + 1;
                    if distArray(i+1) > maxPositive
                        maxPositive = distArray(i+1);
                        max1 = i+1;
                    end
                elseif distArray(i+1) < 0
                    negativeCount = negativeCount + 1;  
                    if abs(distArray(i+1)) > abs(maxNegative)
                        maxNegative = distArray(i+1);
                        max2 = i+1;
                    end
                end
            end

            if i + 1 - t >= 2
                if i + 1 - t < 6
                    maxspeed = maxspeed + 1;
                end    
                if sign(maxPositive) * sign(distArray(t)) == 1 && (positiveCount >= negativeCount)
                    bendnum = bendnum + 1;
                elseif sign(maxNegative) * sign(distArray(t)) == 1 && (negativeCount >= positiveCount)
                    bendnum = bendnum + 1;
                end
            end
            i = i + 1; 
        end      
    end
end
function [bendnum] = analyzeHeadBends(distArray,   trainPath, bendnum, wlength)
    t = 1;
    i = 2;
    [positiveCount, negativeCount, maxPositive, maxNegative, max1, max2] = deal(0, 0, 0, 0, 0, 0);
    while i <= length(distArray) - 2
        if sign(distArray(t)) * sign(distArray(i)) == 1 || (sign(distArray(t)) * sign(distArray(i)) == -1 && sign(distArray(i+1)) * sign(distArray(i)) == -1) ||distArray(t) * distArray(i) == 0
            i = i + 1; 
        elseif sign(distArray(t)) * sign(distArray(i)) == -1 && sign(distArray(i+1)) * sign(distArray(i)) == 1
            t = i;
            [positiveCount, negativeCount, maxPositive, maxNegative, max1, max2] = deal(0, 0, 0, 0, 0, 0);

            while i + 2 <= length(distArray)
                if distArray(i) > 0
                    positiveCount = positiveCount + 1;
                    if distArray(i) > maxPositive
                        maxPositive = distArray(i);
                        max1 = i;
                    end
                elseif distArray(i) < 0
                    negativeCount = negativeCount + 1;  
                    if abs(distArray(i)) > abs(maxNegative)
                        maxNegative = distArray(i);
                        max2 = i;
                    end
                end

                if sign(distArray(t)) * sign(distArray(i+1)) == -1
                    if (i + 1 - t > 1) && (sign(distArray(t)) * sign(distArray(i+2)) == -1)
                        break;
                    end
                end
                i = i + 1;    
            end

            if i + 1 == length(distArray)
                if distArray(i) > 0
                    positiveCount = positiveCount + 1;
                    if distArray(i) > maxPositive
                        maxPositive = distArray(i);
                        max1 = i;
                    end
                elseif distArray(i) < 0
                    negativeCount = negativeCount + 1;  
                    if abs(distArray(i)) > abs(maxNegative)
                        maxNegative = distArray(i);
                        max2 = i;
                    end
                end

                if distArray(i+1) > 0
                    positiveCount = positiveCount + 1;
                    if distArray(i+1) > maxPositive
                        maxPositive = distArray(i+1);
                        max1 = i+1;
                    end
                elseif distArray(i+1) < 0
                    negativeCount = negativeCount + 1;  
                    if abs(distArray(i+1)) > abs(maxNegative)
                        maxNegative = distArray(i+1);
                        max2 = i+1;
                    end
                end
            end

            if i + 1 - t >= 2
                if sign(maxPositive) * sign(distArray(t)) == 1 && (positiveCount >= negativeCount)
                    bendnum = bendnum + 1;
                elseif sign(maxNegative) * sign(distArray(t)) == 1 && (negativeCount >= positiveCount)
                    bendnum = bendnum + 1;
                end
            end
            i = i + 1; 
        end      
    end
end

function [bendnum] = analyzeTailBends(distArray, trainPath, bendnum, wlength)
    t = 1;
    i = 2;
    [positiveCount, negativeCount, maxPositive, maxNegative, max1, max2] = deal(0, 0, 0, 0, 0, 0);
    while i <= length(distArray) - 2
        if sign(distArray(t)) * sign(distArray(i)) == 1 || (sign(distArray(t)) * sign(distArray(i)) == -1 && sign(distArray(i+1)) * sign(distArray(i)) == -1) ||distArray(t) * distArray(i) == 0
            i = i + 1; 
        elseif sign(distArray(t)) * sign(distArray(i)) == -1 && sign(distArray(i+1)) * sign(distArray(i)) == 1
            t = i;
            [positiveCount, negativeCount, maxPositive, maxNegative, max1, max2] = deal(0, 0, 0, 0, 0, 0);

            while i + 2 <= length(distArray)
                if distArray(i) > 0
                    positiveCount = positiveCount + 1;
                    if distArray(i) > maxPositive
                        maxPositive = distArray(i);
                        max1 = i;
                    end
                elseif distArray(i) < 0
                    negativeCount = negativeCount + 1;  
                    if abs(distArray(i)) > abs(maxNegative)
                        maxNegative = distArray(i);
                        max2 = i;
                    end
                end

                if sign(distArray(t)) * sign(distArray(i+1)) == -1
                    if (i + 1 - t > 1) && (sign(distArray(t)) * sign(distArray(i+2)) == -1)
                        break;
                    end
                end
                i = i + 1;    
            end

            if i + 1 == length(distArray)
                if distArray(i) > 0
                    positiveCount = positiveCount + 1;
                    if distArray(i) > maxPositive
                        maxPositive = distArray(i);
                        max1 = i;
                    end
                elseif distArray(i) < 0
                    negativeCount = negativeCount + 1;  
                    if abs(distArray(i)) > abs(maxNegative)
                        maxNegative = distArray(i);
                        max2 = i;
                    end
                end

                if distArray(i+1) > 0
                    positiveCount = positiveCount + 1;
                    if distArray(i+1) > maxPositive
                        maxPositive = distArray(i+1);
                        max1 = i+1;
                    end
                elseif distArray(i+1) < 0
                    negativeCount = negativeCount + 1;  
                    if abs(distArray(i+1)) > abs(maxNegative)
                        maxNegative = distArray(i+1);
                        max2 = i+1;
                    end
                end
            end

            if i + 1 - t >= 2
                if sign(maxPositive) * sign(distArray(t)) == 1 && (positiveCount >= negativeCount)
                    bendnum = bendnum + 1;
                elseif sign(maxNegative) * sign(distArray(t)) == 1 && (negativeCount >= positiveCount)
                    bendnum = bendnum + 1;
                end
            end
            i = i + 1; 
        end      
    end
end
function rowsToKeep = getValidRows(varargin)

    numFiles = length(varargin);
    rowsToKeep = true(size(varargin{1}, 1), 1); 
    
    for i = 1:numFiles
        currentData = varargin{i};

        validRows = ~all(isnan(currentData), 2);
        rowsToKeep = rowsToKeep & validRows;
end

function saveBendResult(filepath, filename)
    fileID = fopen(filepath, 'a');  % 'a' for append mode
    fprintf(fileID, 'Body bend detected in file: %s\n', filename);
    fclose(fileID);
end

function saveToCSV(filePath, data)

    writematrix(data', filePath);
end

function saveFeatures(filePath, wlength, bodyBendnum, headBendnum, tailBendnum, maxspeed, Omega)

    features = {'wrom_id', 'wlength', 'bodyBendnum', 'headBendnum', 'tailBendnum', 'maxspeed', 'Omega';
                1, wlength, bodyBendnum, headBendnum, tailBendnum, maxspeed, Omega};

    writecell(features, filePath, 'Delimiter', ',');
end

