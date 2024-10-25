# WormYOLO(Image segmentation algorithm)
### Steps

1.Place the images to be processed into the subfolder "worm_data" within the "worm_seg" folder.

2. Obtain the necessary model weight files. 
swin_large_patch4_window12_384_22k.pth from Swin backbone weights is required. You can choose between mate_worm.pth and multi_worm.pth, where mate_worm.pth is more suitable for mating scenarios and multi_worm.pth for general multi-worm scenarios. Weights are available from 'https://zenodo.org/records/13234171'.

3. Place the required weight files into the "worm_seg" folder and then place the "worm_seg" folder into the root directory of the installed mmdetection.

4. Run the ‘segmentation.py’ file. A new folder will be automatically created within the "worm_data" folder, and the segmented images will be saved inside it.

# Feature points extraction
### Steps
1. Place the preprocessed image set into the subfolder named "samples" within the "feature points extraction" folder.

2. Double click to open ‘feature points extraction.sln’.

3. Run the program and obtain the coordinates of the pharynx, inflection point, peak point, and skeletal point. These results are saved in the files below.

# Count
### Steps
1. Please place the 5 ".csv" files obtained from the feature point detection algorithm into the "Feature_point" subfolder within the "Tracker" folder.

2. Open the program using Matlab R2021a, run ‘Tracker_ Feature.p’, and obtain the experimental results.



