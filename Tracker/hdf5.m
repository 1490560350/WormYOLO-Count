function video_to_hdf5(videoPath, hdf5Path)
    % 创建VideoReader对象读取视频文件
    videoObj = VideoReader(videoPath);
    
    % 获取视频的帧率和总帧数
    fps = videoObj.FrameRate;
    totalFrames = videoObj.NumFrames;
    
    % 打开HDF5文件进行写入
    h5create(hdf5Path, '/frames', [totalFrames, videoObj.Height, videoObj.Width, 3], 'Datatype', 'uint8');
    
    % 循环读取并保存每一帧
    for frameNum = 1:totalFrames
        frame = readFrame(videoObj);
        
        % 将帧数据写入HDF5文件
        h5write(hdf5Path, '/frames', frame, [frameNum, 1, 1, 1], [1, videoObj.Height, videoObj.Width, 3]);
    end
    
    fprintf('视频已分帧并保存到 %s\n', hdf5Path);
end

% 使用示例
videoPath = 'E:\data3\tierpsy-tracker\data\AVI_VIDEOS\AVI_VIDEOS_1.avi';
hdf5Path = 'E:\data3\tierpsy-tracker\data\AVI_VIDEOS\Results\AVI_VIDEOS_1_featuresN.hdf5';
video_to_hdf5(videoPath, hdf5Path);
