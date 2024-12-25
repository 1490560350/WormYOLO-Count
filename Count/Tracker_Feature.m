clc;
close all;
clear all;
trainPath = fileparts(mfilename('fullpath'));
dataPath = fullfile(trainPath, 'Feature_point');
HT = xlsread(fullfile(dataPath, 'HeadTailReg.csv'));
Ph = xlsread(fullfile(dataPath, 'Pharynx.csv'));
PP = xlsread(fullfile(dataPath, 'PeakPoints.csv'));
IP = xlsread(fullfile(dataPath, 'InflectionPoints.csv'));
SC = xlsread(fullfile(dataPath, 'SkeletonPoints.csv'));
rowsToKeep = getValidRows(HT, Ph, PP, IP, SC);
HT = HT(rowsToKeep, :);
Ph = Ph(rowsToKeep, :);
PP = PP(rowsToKeep, :);
IP = IP(rowsToKeep, :);
SC = SC(rowsToKeep, :);

deletedRows = find(~rowsToKeep);
if ~isempty(deletedRows)
    fprintf('In the input files, if any file has missing data in one or more rows, the corresponding rows in the other files have been deleted. The remaining data is not affected.\n');
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

[bodyBendnum, headBendnum, tailBendnum, maxspeed] = analyzeBends(maxDist, maxHDist, maxTDist,  trainPath, bodyBendnum, headBendnum, tailBendnum, wlength);
fprintf('Dist: the number of Head bends are %d\n', headBendnum);
fprintf('Dist: the number of Body bends are %d\n', bodyBendnum);
fprintf('Dist: the number of Tail bends are %d\n', tailBendnum);
saveFeatures(fullfile(dataPath, 'Features.csv'), wlength, bodyBendnum, headBendnum, tailBendnum, maxspeed, Omega);
