clc;
close all;
clear all;

trainPath = 'E:\data3\daf3\20100722T114524\out\';
savePath = fullfile(trainPath, 'pictuer');
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

theFiles  = dir([trainPath '*.png']);
disp(length(theFiles));
count = 1;
train_num = length(theFiles);
sort_nat_name = sort_nat({theFiles.name}); 

% 读取数据
HT = xlsread(fullfile(trainPath, 'analysis result', 'HeadTailReg.csv'));
Ph = xlsread(fullfile(trainPath, 'analysis result', 'Pharynx.csv'));
PP = xlsread(fullfile(trainPath, 'analysis result', 'PeakPoints.csv'));
IP = xlsread(fullfile(trainPath, 'analysis result', 'InflectionPoints.csv'));
SC = xlsread(fullfile(trainPath, 'analysis result', 'Center.csv'));
numRows = size(HT, 1); 

% 计算长度
wlength = Lengthcalculation(SC);

% Initialize arrays
head = HT(:,1:2);
headt = SC(:,41:42);
tail = HT(:,3:4);
tailh = SC(:,201:202);
pharynx = Ph(:,1:2);


maxHDist = [];
maxTDist = [];
bodyBendnum = 0;
headBendnum = 0;
tailBendnum = 0;
maxspeed = 0;
Omega = 0;

for k = 1:train_num

 
    hdists = [];
    tdists = [];
    for j = 3:2:39

        hflag = bendDirection(head(k,1), head(k,2), headt(k,1), headt(k,2), SC(k,j), SC(k,j+1)); 
        hdist = Dist(SC(k,j), SC(k,j+1), head(k,1), head(k,2), headt(k,1), headt(k,2)); 
        hdist = hdist * hflag;
        hdists = [hdists hdist];       
    end
    
    if isempty(hdists)
        continue; % 跳过当前文件，处理下一个文件
    end

    maxHDistValue = max(abs(hdists));
    index2 = find(abs(hdists) == maxHDistValue, 1);
    maxHDist = [maxHDist hdists(index2)];
    
    for z = 203:2:239
        tflag = bendDirection( tailh(k,1),  tailh(k,2),  tail(k,1),  tail(k,2), SC(k,z), SC(k,z+1)); 
        tdist = Dist(SC(k,z), SC(k,z+1), tailh(k,1), tailh(k,2), tail(k,1), tail(k,2)); 
        tdist = tdist * tflag;
        tdists = [tdists tdist];     
    end
    if isempty(tdists)
        continue; % 跳过当前文件，处理下一个文件
    end
   
    maxTDistValue = max(abs(tdists));
    index = find(abs(tdists) == maxTDistValue, 1);
    index = index;
    maxTDist = [maxTDist tdists(index)];
    % 加载图像
    imagePath = fullfile(trainPath, sort_nat_name{k});
    img = imread(imagePath);
    figure, imshow(img);
    hold on;

    % 绘制head点
    plot(head(k,1), head(k,2), 'ro', 'MarkerSize', 3, 'LineWidth', 2);
    plot(headt(k,1), headt(k,2), 'bo', 'MarkerSize', 3, 'LineWidth', 2);
    plot(tailh(k,1), tailh(k,2), 'ro', 'MarkerSize', 3, 'LineWidth', 2);
    plot(tail(k,1), tail(k,2), 'ro', 'MarkerSize', 3, 'LineWidth', 2);
    % 绘制maxDist点
    maxHDistX = SC(k, index2*2 - 1);
    maxHDistY = SC(k, index2*2);

    plot(maxHDistX, maxHDistY, 'go', 'MarkerSize', 3, 'LineWidth', 2);
     % 连接Head和HeadT的直线
    line([head(k,1), headt(k,1)], [head(k,2), headt(k,2)], 'Color', 'r', 'LineWidth', 2);
    maxTDistX = SC(k, (index+101)*2-1);
    maxTDistY = SC(k, (index+101)*2);
    plot(maxTDistX, maxTDistY, 'o', 'MarkerSize', 3, 'LineWidth', 2);
     % 连接Head和HeadT的直线
    line([tailh(k,1),tail(k,1)], [tailh(k,2), tail(k,2)], 'Color', 'r', 'LineWidth', 2);
    % 添加图例
    legend('Head', 'HeadT', 'MaxDist Point');
    hold off;

    % 保存图像到pictuer文件夹
    saveas(gcf, fullfile(savePath, sprintf('result_%d.png', k)));
    close all;

    % 处理 IP 数据
    IPnum = sum(~isnan(IP(k, :)));
    % 根据需要处理 IP 数据
end
